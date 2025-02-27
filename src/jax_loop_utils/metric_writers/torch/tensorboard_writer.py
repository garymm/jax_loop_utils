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

"""MetricWriter for Pytorch summary files.

Use this writer for the Pytorch-based code.

"""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from absl import logging
from torch.utils.tensorboard.writer import SummaryWriter

from jax_loop_utils.metric_writers import interface

Array = interface.Array
Scalar = interface.Scalar

try:
    from moviepy import editor as mpy
except ImportError:
    mpy = None


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
        # Note: moviepy is used internally by torch.utils.tensorboard.summary.make_video.
        if mpy is None:
            logging.log_first_n(
                logging.WARNING,
                "torch.TensorboardWriter.write_videos requires moviepy to be installed.",
                1,
            )
            return
        for key, value in videos.items():
            if value.ndim != 4 or value.shape[-1] not in (1, 3):
                raise ValueError(
                    "Expected an array with shape (T, H, W, 1) or (T, H, W, 3)."
                    f"Got shape {value.shape} with dtype {value.dtype}."
                )
            value = np.transpose(value, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            value = np.expand_dims(value, axis=0)  # add batch dimension, expected by add_video
            self._writer.add_video(key, value, global_step=step)

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
