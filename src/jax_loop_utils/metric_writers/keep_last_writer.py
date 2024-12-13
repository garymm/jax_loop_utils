from collections.abc import Mapping
from typing import Any, Optional

from .interface import Array, MetricWriter, Scalar


class KeepLastWriter(MetricWriter):
    """MetricWriter that keeps the last value for each metric in memory."""

    def __init__(self, inner: MetricWriter):
        self._inner: MetricWriter = inner
        self.scalars: Optional[Mapping[str, Scalar]] = None
        self.images: Optional[Mapping[str, Array]] = None
        self.videos: Optional[Mapping[str, Array]] = None
        self.audios: Optional[Mapping[str, Array]] = None
        self.texts: Optional[Mapping[str, str]] = None
        self.hparams: Optional[Mapping[str, Any]] = None
        self.histogram_arrays: Optional[Mapping[str, Array]] = None
        self.histogram_num_buckets: Optional[Mapping[str, int]] = None

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        self._inner.write_scalars(step, scalars)
        self.scalars = scalars

    def write_images(self, step: int, images: Mapping[str, Array]):
        self._inner.write_images(step, images)
        self.images = images

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        self._inner.write_videos(step, videos)
        self.videos = videos

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        self._inner.write_audios(step, audios, sample_rate=sample_rate)
        self.audios = audios

    def write_texts(self, step: int, texts: Mapping[str, str]):
        self._inner.write_texts(step, texts)
        self.texts = texts

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._inner.write_hparams(hparams)
        self.hparams = hparams

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        self._inner.write_histograms(step, arrays, num_buckets)
        self.histogram_arrays = arrays
        self.histogram_num_buckets = num_buckets
