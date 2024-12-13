"""Tests for PrefixSuffixWriter."""

from unittest import mock

import numpy as np
from absl.testing import absltest

from jax_loop_utils.metric_writers import memory_writer, prefix_suffix_writer


class PrefixSuffixWriterTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.memory_writer = memory_writer.MemoryWriter()
        self.writer = prefix_suffix_writer.PrefixSuffixWriter(
            self.memory_writer,
            prefix="prefix/",
            suffix="/suffix",
        )

    def test_write_scalars(self):
        self.writer.write_scalars(0, {"metric": 1.0})
        self.assertEqual(self.memory_writer.scalars, {0: {"prefix/metric/suffix": 1.0}})

    def test_write_images(self):
        image = np.zeros((2, 2, 3))
        self.writer.write_images(0, {"image": image})
        self.assertEqual(
            list(self.memory_writer.images[0].keys()), ["prefix/image/suffix"]
        )

    def test_write_texts(self):
        self.writer.write_texts(0, {"text": "hello"})
        self.assertEqual(self.memory_writer.texts, {0: {"prefix/text/suffix": "hello"}})

    def test_write_histograms(self):
        data = np.array([1, 2, 3])
        buckets = {"hist": 10}
        self.writer.write_histograms(0, {"hist": data}, buckets)
        self.assertEqual(
            list(self.memory_writer.histograms[0].arrays.keys()),
            ["prefix/hist/suffix"],
        )

    def test_write_hparams(self):
        self.writer.write_hparams({"param": 1})
        self.assertEqual(self.memory_writer.hparams, {"prefix/param/suffix": 1})

    def test_empty_prefix_suffix(self):
        writer = prefix_suffix_writer.PrefixSuffixWriter(self.memory_writer)
        writer.write_scalars(0, {"metric": 1.0})
        self.assertEqual(self.memory_writer.scalars, {0: {"metric": 1.0}})

    def test_write_videos(self):
        video = np.zeros((10, 32, 32, 3))  # Simple video array with 10 frames
        self.writer.write_videos(0, {"video": video})
        self.assertEqual(
            list(self.memory_writer.videos[0].keys()), ["prefix/video/suffix"]
        )

    def test_write_audios(self):
        audio = np.zeros((16000,))  # 1 second of audio at 16kHz
        self.writer.write_audios(0, {"audio": audio}, sample_rate=16000)
        self.assertEqual(
            list(self.memory_writer.audios[0].audios.keys()), ["prefix/audio/suffix"]
        )

    def test_close(self):
        with mock.patch.object(self.memory_writer, "close") as mock_close:
            self.writer.close()
            mock_close.assert_called_once()

    def test_flush(self):
        with mock.patch.object(self.memory_writer, "flush") as mock_flush:
            self.writer.flush()
            mock_flush.assert_called_once()


if __name__ == "__main__":
    absltest.main()
