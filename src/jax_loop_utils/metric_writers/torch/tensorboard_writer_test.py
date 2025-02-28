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

"""Tests for TensorboardWriter."""

import collections
import os
from typing import Any

import numpy as np
import tensorflow as tf

from jax_loop_utils.metric_writers.torch import tensorboard_writer


def _load_scalars_data(logdir: str):
    """Loads scalar summaries from events in a logdir."""
    paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
    data = collections.defaultdict(dict)
    for path in paths:
        for event in tf.compat.v1.train.summary_iterator(path):
            for value in event.summary.value:
                assert value.HasField("tensor")
                assert len(value.tensor.float_val) == 1
                data[event.step][value.tag] = value.tensor.float_val[0]

    return data


def _load_histograms_data(logdir: str) -> dict[int, dict[str, Any]]:
    """Loads histograms summaries from events in a logdir.

    Args:
      logdir: a directory to find logs

    Returns:
      A generated histograms in a shape step -> tag -> histo.
    """
    paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
    data = {}
    for path in paths:
        for event in tf.compat.v1.train.summary_iterator(path):
            if event.step not in data:
                data[event.step] = {}
            step_data = {}
            for value in event.summary.value:
                step_data[value.tag] = value.histo
            data[event.step].update(step_data)

    return data


def _load_images_data(logdir: str) -> dict[int, dict[str, Any]]:
    """Loads image summaries from events in a logdir."""
    paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
    data = collections.defaultdict(dict)
    for path in paths:
        for event in tf.compat.v1.train.summary_iterator(path):
            for value in event.summary.value:
                if value.HasField("image"):
                    data[event.step][value.tag] = value.image

    return data


class TensorboardWriterTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.logdir = self.get_temp_dir()
        self.writer = tensorboard_writer.TensorboardWriter(self.logdir)

    def test_write_scalar(self):
        self.writer.write_scalars(11, {"a": 0.6, "b": 15})
        self.writer.write_scalars(20, {"a": 0.8, "b": 12})
        self.writer.flush()
        data = _load_scalars_data(self.logdir)
        self.assertAllClose(data[11], {"a": 0.6, "b": 15})
        self.assertAllClose(data[20], {"a": 0.8, "b": 12})

    def test_write_histograms(self):
        self.writer.write_histograms(
            0,
            {
                "a": np.asarray([0.3, 0.1, 0.5, 0.7, 0.1]),
                "b": np.asarray([-0.1, 0.3, 0.2, 0.4, 0.4]),
            },
            num_buckets={"a": 2, "b": 2},
        )
        self.writer.write_histograms(
            2,
            {
                "a": np.asarray([0.2, 0.4, 0.5, 0.1, -0.1]),
                "b": np.asarray([0.7, 0.3, 0.2, 0.1, 0.0]),
            },
            num_buckets={"a": 2, "b": 2},
        )
        self.writer.flush()
        data = _load_histograms_data(self.logdir)
        self.assertNear(data[0]["a"].min, 0.1, 0.001)
        self.assertNear(data[0]["a"].max, 0.7, 0.001)
        self.assertNear(data[0]["b"].min, -0.1, 0.001)
        self.assertNear(data[0]["b"].max, 0.4, 0.001)
        self.assertNear(data[2]["a"].min, -0.1, 0.001)
        self.assertNear(data[2]["a"].max, 0.5, 0.001)
        self.assertNear(data[2]["b"].min, 0.0, 0.001)
        self.assertNear(data[2]["b"].max, 0.7, 0.001)

    def test_write_videos(self):
        # Generate 100 frames of noise video
        frames = []
        for _ in range(100):
            frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            frames.append(frame)

        videos = {
            "zzz/noise_0": np.stack(frames, axis=0),
            "noise_1": np.stack(frames, axis=0),
        }
        self.writer.write_videos(0, videos)
        self.writer.close()
        data = _load_images_data(self.logdir)
        self.assertIn("zzz/noise_0", data[0])
        self.assertIn("noise_1", data[0])
        self.assertEqual(data[0]["zzz/noise_0"].height, 64)
        self.assertEqual(data[0]["zzz/noise_0"].width, 64)
        self.assertEqual(data[0]["noise_1"].height, 64)
        self.assertEqual(data[0]["noise_1"].width, 64)


if __name__ == "__main__":
    tf.test.main()
