import tempfile

import jax.numpy as jnp
import mlflow
import mlflow.entities
import mlflow.exceptions
import mlflow.protos.databricks_pb2
import numpy as np
from absl.testing import absltest

from jax_loop_utils.metric_writers.mlflow import MlflowMetricWriter


def _get_runs(tracking_uri: str, experiment_name: str) -> list[mlflow.entities.Run]:
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    return client.search_runs([experiment.experiment_id])


def _exceptional_mlflow_client_class(
    actually_create: bool,
) -> type[mlflow.MlflowClient]:
    class ExceptionalMlflowClient(mlflow.MlflowClient):
        def create_experiment(self, *args, **kwargs):
            if actually_create:
                super().create_experiment(*args, **kwargs)
            raise mlflow.exceptions.MlflowException(
                "Experiment already exists",
                error_code=mlflow.protos.databricks_pb2.RESOURCE_ALREADY_EXISTS,
            )

    return ExceptionalMlflowClient


class MlflowMetricWriterTest(absltest.TestCase):
    def test_set_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            writer.write_tags({"ooh": "aah"})
            writer.flush()
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.data.tags["ooh"], "aah")

    def test_write_scalars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            seq_of_scalars = (
                {"a": 3, "b": 0.15},
                {"a": 5, "b": 0.007},
            )
            for step, scalars in enumerate(seq_of_scalars):
                writer.write_scalars(step, scalars)
            writer.flush()
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            for metric_key in ("a", "b"):
                self.assertIn(metric_key, run.data.metrics)
                self.assertEqual(run.data.metrics[metric_key], seq_of_scalars[-1][metric_key])
            # constant defined in mlflow.entities.RunStatus
            self.assertEqual(run.info.status, "RUNNING")
            writer.close()
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.info.status, "FINISHED")
            # check we can create a new writer with an existing experiment
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            writer.write_scalars(0, {"a": 1, "b": 2})
            writer.flush()
            writer.close()
            # should result in a new run
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 2)

    def test_write_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            writer.write_images(0, {"test_image": jnp.zeros((3, 3, 3), dtype=np.uint8)})
            writer.close()

            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            # the string "images" is hardcoded in MlflowClient.log_image.
            artifacts = writer._client.list_artifacts(run.info.run_id, "images")
            artifact_paths = [artifact.path for artifact in artifacts]
            self.assertGreaterEqual(len(artifact_paths), 1)
            self.assertIn("test_image", artifact_paths[0])

    def test_write_texts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            test_text = "Hello world"
            writer.write_texts(0, {"test_text": test_text})
            writer.close()

            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            artifacts = writer._client.list_artifacts(run.info.run_id)
            artifact_paths = [artifact.path for artifact in artifacts]
            self.assertGreaterEqual(len(artifact_paths), 1)
            self.assertIn("test_text_step_0.txt", artifact_paths)
            local_path = writer._client.download_artifacts(run.info.run_id, "test_text_step_0.txt")
            with open(local_path) as f:
                content = f.read()
            self.assertEqual(content, test_text)

    def test_write_hparams(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            test_params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
            writer.write_hparams(test_params)
            writer.close()

            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.data.params["learning_rate"], "0.001")
            self.assertEqual(run.data.params["batch_size"], "32")
            self.assertEqual(run.data.params["epochs"], "100")

    def test_write_videos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)

            # Generate 100 frames of noise video
            frames = []
            for _ in range(100):
                frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                frames.append(frame)

            videos = {
                "zzz/noise_0": np.stack(frames, axis=0),
                "noise_1": np.stack(frames, axis=0),
            }
            writer.write_videos(0, videos)
            writer.close()

            # Verify artifacts were written
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]

            artifacts_videos = writer._client.list_artifacts(run.info.run_id, "videos")
            self.assertEqual(len(artifacts_videos), 2)
            sorted_artifacts_videos = sorted(artifacts_videos, key=lambda x: x.path)
            self.assertEqual(sorted_artifacts_videos[0].path, "videos/noise_1_000000000.mp4")
            self.assertFalse(sorted_artifacts_videos[0].is_dir)

            artifacts_zzz = writer._client.list_artifacts(run.info.run_id, "videos/zzz")
            self.assertEqual(len(artifacts_zzz), 1)
            self.assertEqual(artifacts_zzz[0].path, "videos/zzz/noise_0_000000000.mp4")
            self.assertFalse(artifacts_zzz[0].is_dir)

    def test_no_ops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "experiment_name"
            writer = MlflowMetricWriter(experiment_name, tracking_uri=tracking_uri)
            writer.write_audios(0, {"audio": np.zeros((2, 1000))}, sample_rate=16000)
            writer.write_histograms(
                0, {"histogram": np.zeros((10,))}, num_buckets={"histogram": 10}
            )
            writer.close()
            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            artifacts = writer._client.list_artifacts(run.info.run_id)
            # the above ops are all no-ops so no artifacts, metrics or params
            self.assertEqual(len(artifacts), 0)
            self.assertEqual(run.data.metrics, {})
            self.assertEqual(run.data.params, {})

    def test_experiment_creation_race_condition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "race_condition_experiment"

            writer = MlflowMetricWriter(
                experiment_name,
                tracking_uri=tracking_uri,
                _client_class=_exceptional_mlflow_client_class(True),
            )

            runs = _get_runs(tracking_uri, experiment_name)
            self.assertEqual(len(runs), 1)
            writer.close()

    def test_experiment_creation_race_condition_and_then_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking_uri = f"file://{temp_dir}"
            experiment_name = "race_condition_experiment"

            self.assertRaises(
                RuntimeError,
                MlflowMetricWriter,
                experiment_name,
                tracking_uri=tracking_uri,
                _client_class=_exceptional_mlflow_client_class(False),
            )


if __name__ == "__main__":
    absltest.main()
