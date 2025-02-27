"""MLflow implementation of MetricWriter interface."""

import os
import pathlib
import shutil
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import mlflow
import mlflow.config
import mlflow.entities
import mlflow.exceptions
import mlflow.protos.databricks_pb2
import mlflow.tracking.fluent
import numpy as np
from absl import logging
from mlflow.entities import RunTag

from jax_loop_utils import asynclib
from jax_loop_utils.metric_writers.interface import (
    Array,
    MetricWriter,
    Scalar,
)

try:
    from jax_loop_utils.metric_writers import _audio_video
except ImportError:
    _audio_video = None


def _noop_decorator(func):
    return func


class MlflowMetricWriter(MetricWriter):
    """Writes metrics to MLflow Tracking."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        _client_class: type[mlflow.MlflowClient] = mlflow.MlflowClient,
    ):
        """Initialize MLflow writer.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run.
            tracking_uri: Address of local or remote tracking server.
              Treated the same as arguments to mlflow.set_tracking_uri.
              See https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
            _client_class: MLflow client class (for testing only).
        """
        self._client = _client_class(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            logging.info(
                "Experiment '%s' does not exist. Creating a new experiment.",
                experiment_name,
            )
            try:
                experiment_id = self._client.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException as e:
                # Handle race in creating experiment.
                if e.error_code != mlflow.protos.databricks_pb2.ErrorCode.Name(
                    mlflow.protos.databricks_pb2.RESOURCE_ALREADY_EXISTS
                ):
                    raise
                experiment = self._client.get_experiment_by_name(experiment_name)
                if not experiment:
                    raise RuntimeError(
                        "Failed to get, then failed to create, "
                        f"then failed to get again experiment '{experiment_name}'"
                    ) from None
                experiment_id = experiment.experiment_id
        self._run_id = self._client.create_run(
            experiment_id=experiment_id, run_name=run_name
        ).info.run_id

    def write_tags(self, tags: dict[str, Any]):
        """Set tags on the MLFlow run"""
        self._client.log_batch(
            self._run_id, [], [], [RunTag(k, str(v)) for k, v in tags.items()], synchronous=False
        )

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar metrics to MLflow."""
        timestamp = int(time.time() * 1000)
        metrics_list = [
            mlflow.entities.Metric(k, float(v), timestamp, step) for k, v in scalars.items()
        ]
        self._client.log_batch(self._run_id, metrics=metrics_list, synchronous=False)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images to MLflow."""
        for key, image_array in images.items():
            self._client.log_image(
                self._run_id,
                np.array(image_array),
                key=key,
                step=step,
                synchronous=False,
            )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Convert videos to files and write them to MLflow.

        Requires the `audio-video` extra to be installed.
        """
        if _audio_video is None:
            logging.log_first_n(
                logging.WARNING,
                "MlflowMetricWriter.write_videos requires the [audio-video] extra to be installed.",
                1,
            )
            return

        pool = asynclib.Pool()

        if len(videos) > 1:
            maybe_async = pool
        else:
            maybe_async = _noop_decorator

        encode_and_log = maybe_async(self._encode_and_log_video)

        paths_arrays = [
            (
                f"{key}_{step:09d}.{_audio_video.CONTAINER_FORMAT}",
                video_array,
            )
            for key, video_array in videos.items()
        ]

        temp_dir = pathlib.Path(tempfile.mkdtemp())
        for path, video_array in paths_arrays:
            encode_and_log(temp_dir, path, video_array)

        pool.close()
        shutil.rmtree(temp_dir)

    def _encode_and_log_video(self, temp_dir: pathlib.Path, rel_path: str, video_array: Array):
        temp_path = temp_dir / rel_path
        # handle keys with slashes
        if not temp_path.parent.exists():
            temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            _audio_video.encode_video(video_array, f)  # pyright: ignore[reportOptionalMemberAccess]
        dest_dir = os.path.join("videos", os.path.dirname(rel_path)).rstrip("/")
        # If log_artifact(synchronous=False) existed,
        # we could synchronize with self.flush() rather than at the end of write_videos.
        # https://github.com/mlflow/mlflow/issues/14153
        self._client.log_artifact(self._run_id, temp_path, dest_dir)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        """MLflow doesn't support audio logging directly."""
        # this could be supported if we convert the audio to a file
        # and log the file as an artifact.
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing audios.",
            1,
        )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        """Write text artifacts to MLflow."""
        for key, text in texts.items():
            self._client.log_text(self._run_id, text, f"{key}_step_{step}.txt")

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        """MLflow doesn't support histogram logging directly.

        https://github.com/mlflow/mlflow/issues/8145
        """
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing histograms.",
            1,
        )

    def write_hparams(self, hparams: Mapping[str, Any]):
        """Log hyperparameters to MLflow."""
        params = [mlflow.entities.Param(key, str(value)) for key, value in hparams.items()]
        self._client.log_batch(self._run_id, params=params, synchronous=False)

    def flush(self):
        """Flushes all logged data."""
        # using private APIs because the public APIs require global state
        # for the current tracking URI and Run ID, and we don't want to
        # create a global state.
        artifact_repo = mlflow.tracking._get_artifact_repo(self._run_id)
        if artifact_repo:
            artifact_repo.flush_async_logging()
        self._client._tracking_client.store.flush_async_logging()

    def close(self):
        """End the MLflow run."""
        self._client.set_terminated(self._run_id)
        self.flush()
