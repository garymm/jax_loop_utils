"""Writer that adds prefix and suffix to metric keys."""

from collections.abc import Mapping
from typing import Any, Optional

from jax_loop_utils.metric_writers import interface


class PrefixSuffixWriter(interface.MetricWriter):
    """Wraps a MetricWriter and adds prefix/suffix to all keys."""

    def __init__(
        self,
        writer: interface.MetricWriter,
        prefix: str = "",
        suffix: str = "",
    ):
        """Initialize the writer.

        Args:
            writer: The underlying MetricWriter to wrap
            prefix: String to prepend to all keys
            suffix: String to append to all keys
        """
        self._writer = writer
        self._prefix = prefix
        self._suffix = suffix

    def _transform_keys(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Add prefix and suffix to all keys in the mapping."""
        return {f"{self._prefix}{key}{self._suffix}": value for key, value in data.items()}

    def write_scalars(self, step: int, scalars: Mapping[str, interface.Scalar]):
        self._writer.write_scalars(step, self._transform_keys(scalars))

    def write_images(self, step: int, images: Mapping[str, interface.Array]):
        self._writer.write_images(step, self._transform_keys(images))

    def write_videos(self, step: int, videos: Mapping[str, interface.Array]):
        self._writer.write_videos(step, self._transform_keys(videos))

    def write_audios(self, step: int, audios: Mapping[str, interface.Array], *, sample_rate: int):
        self._writer.write_audios(step, self._transform_keys(audios), sample_rate=sample_rate)

    def write_texts(self, step: int, texts: Mapping[str, str]):
        self._writer.write_texts(step, self._transform_keys(texts))

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, interface.Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        if num_buckets is not None:
            num_buckets = self._transform_keys(num_buckets)
        self._writer.write_histograms(step, self._transform_keys(arrays), num_buckets)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._writer.write_hparams(self._transform_keys(hparams))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
