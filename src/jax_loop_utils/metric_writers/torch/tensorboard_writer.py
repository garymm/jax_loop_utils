# Copyright 2024 The CLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetricWriter that uses PyTorch's SummaryWriter."""

import io
from collections.abc import Mapping
from typing import Any, Optional

from absl import logging
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard.writer import SummaryWriter

from jax_loop_utils import asynclib
from jax_loop_utils.metric_writers import interface

Array = interface.Array
Scalar = interface.Scalar

try:
    from jax_loop_utils.metric_writers import _audio_video
except ImportError:
    _audio_video = None


def _noop_decorator(func):
    return func


class TensorboardWriter(interface.MetricWriter):
    """MetricWriter that writes Pytorch summary files."""

    def __init__(self, logdir: str):
        super().__init__()
        self._writer = SummaryWriter(log_dir=logdir)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, global_step=step, new_style=True)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self._writer.add_image(key, value, global_step=step, dataformats="HWC")

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Convert videos to GIFs and write them to Tensorboard.

        Requires the `audio-video` extra to be installed.
        """
        if _audio_video is None:
            logging.log_first_n(
                logging.WARNING,
                "MlflowMetricWriter.write_videos requires the [audio-video] extra to be installed.",
                1,
            )
            return
        # NOTE: not using self._writer.add_video because
        # https://github.com/pytorch/pytorch/issues/147317
        pool = asynclib.Pool()

        if len(videos) > 1:
            maybe_async = pool
        else:
            maybe_async = _noop_decorator

        encode_and_log = maybe_async(self._encode_and_log_video)

        for key, video_array in videos.items():
            encode_and_log(key, video_array, step)

        pool.close()

    def _encode_and_log_video(self, key: str, video_array: Array, step: int):
        f = io.BytesIO()
        _audio_video.encode_video_to_gif(video_array, f)  # pyright: ignore[reportOptionalMemberAccess]
        image = Summary.Image(  # pyright: ignore[reportAttributeAccessIssue]
            height=video_array.shape[1],
            width=video_array.shape[2],
            colorspace=video_array.shape[3],
            encoded_image_string=f.getvalue(),
        )
        f.close()
        summary = Summary(value=[Summary.Value(tag=key, image=image)])  # pyright: ignore[reportCallIssue,reportAttributeAccessIssue]
        self._writer._get_file_writer().add_summary(summary, step, None)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        for key, value in audios.items():
            self._writer.add_audio(key, value, global_step=step, sample_rate=sample_rate)

    def write_texts(self, step: int, texts: Mapping[str, str]):
        raise NotImplementedError("torch.TensorboardWriter does not support writing texts.")

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Optional[Mapping[str, int]] = None,
    ):
        for tag, values in arrays.items():
            bins = None if num_buckets is None else num_buckets.get(tag)
            self._writer.add_histogram(tag, values, global_step=step, bins="auto", max_bins=bins)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._writer.add_hparams(hparams, {})

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
