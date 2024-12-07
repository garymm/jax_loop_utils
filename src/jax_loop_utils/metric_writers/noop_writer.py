"""No-op MetricWriter implementation."""

from collections.abc import Mapping
from typing import Any, Optional

from jax_loop_utils.metric_writers.interface import Array, MetricWriter, Scalar


class NoOpWriter(MetricWriter):
    """MetricWriter that performs no operations."""

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        pass

    def write_images(self, step: int, images: Mapping[str, Array]):
        pass

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        pass

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        pass

    def write_texts(self, step: int, texts: Mapping[str, str]):
        pass

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        pass

    def write_hparams(self, hparams: Mapping[str, Any]):
        pass

    def flush(self):
        pass

    def close(self):
        pass
