"""MLflow implementation of MetricWriter interface."""

from collections.abc import Mapping
from time import time
from typing import Any

import mlflow
import mlflow.config
import mlflow.entities
import mlflow.tracking.fluent
from absl import logging

from jax_loop_utils.metric_writers.interface import (
    Array,
    Scalar,
)
from jax_loop_utils.metric_writers.interface import (
    MetricWriter as MetricWriterInterface,
)


class MlflowMetricWriter(MetricWriterInterface):
    """Writes metrics to MLflow Tracking."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow writer.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run.
            tracking_uri: Address of local or remote tracking server.
              Treated the same as arguments to mlflow.set_tracking_uri.
              See https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
        """
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            logging.info(
                "Experiment with name '%s' does not exist. Creating a new experiment.",
                experiment_name,
            )
            experiment_id = self._client.create_experiment(experiment_name)
        self._run_id = self._client.create_run(
            experiment_id=experiment_id, run_name=run_name
        ).info.run_id

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar metrics to MLflow."""
        timestamp = int(time() * 1000)
        metrics_list = [
            mlflow.entities.Metric(k, float(v), timestamp, step)
            for k, v in scalars.items()
        ]
        self._client.log_batch(self._run_id, metrics=metrics_list, synchronous=False)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images to MLflow."""
        for key, image_array in images.items():
            self._client.log_image(
                self._run_id, image_array, key=key, step=step, synchronous=False
            )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """MLflow doesn't support video logging directly."""
        # this could be supported if we convert the video to a file
        # and log the file as an artifact.
        logging.log_first_n(
            logging.WARNING,
            "mlflow.MetricWriter does not support writing videos.",
            1,
        )

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        """MLflow doesn't support audio logging directly."""
        # this could be supported if we convert the video to a file
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
        params = [
            mlflow.entities.Param(key, str(value)) for key, value in hparams.items()
        ]
        self._client.log_batch(self._run_id, params=params, synchronous=False)

    def flush(self):
        """Flushes all logged data."""
        mlflow.flush_artifact_async_logging()
        mlflow.flush_async_logging()
        mlflow.flush_trace_async_logging()

    def close(self):
        """End the MLflow run."""
        self._client.set_terminated(self._run_id)
        self.flush()
