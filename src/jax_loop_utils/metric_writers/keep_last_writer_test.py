"""Tests for KeepLastMetricWriter."""

import numpy as np
from absl.testing import absltest

from jax_loop_utils.metric_writers import keep_last_writer, noop_writer


class KeepLastWriterTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.writer = keep_last_writer.KeepLastWriter(noop_writer.NoOpWriter())

    def test_write_scalars(self):
        scalars1 = {"metric1": 1.0}
        scalars2 = {"metric2": 2.0}

        self.writer.write_scalars(0, scalars1)
        self.assertEqual(self.writer.scalars, scalars1)

        self.writer.write_scalars(1, scalars2)
        self.assertEqual(self.writer.scalars, scalars2)

    def test_write_images(self):
        image = np.zeros((2, 2, 3))
        images = {"image": image}

        self.writer.write_images(0, images)
        self.assertEqual(self.writer.images, images)

    def test_write_videos(self):
        video = np.zeros((10, 2, 2, 3))
        videos = {"video": video}

        self.writer.write_videos(0, videos)
        self.assertEqual(self.writer.videos, videos)

    def test_write_audios(self):
        audio = np.zeros((100, 2))
        audios = {"audio": audio}

        self.writer.write_audios(0, audios, sample_rate=44100)
        self.assertEqual(self.writer.audios, audios)

    def test_write_texts(self):
        texts = {"text": "hello world"}

        self.writer.write_texts(0, texts)
        self.assertEqual(self.writer.texts, texts)

    def test_write_hparams(self):
        hparams = {"learning_rate": 0.1}

        self.writer.write_hparams(hparams)
        self.assertEqual(self.writer.hparams, hparams)

    def test_write_histograms(self):
        arrays = {"hist": np.array([1, 2, 3])}
        num_buckets = {"hist": 10}

        self.writer.write_histograms(0, arrays, num_buckets)
        self.assertEqual(self.writer.histogram_arrays, arrays)
        self.assertEqual(self.writer.histogram_num_buckets, num_buckets)

    def test_initial_values_are_none(self):
        self.assertIsNone(self.writer.scalars)
        self.assertIsNone(self.writer.images)
        self.assertIsNone(self.writer.videos)
        self.assertIsNone(self.writer.audios)
        self.assertIsNone(self.writer.texts)
        self.assertIsNone(self.writer.hparams)
        self.assertIsNone(self.writer.histogram_arrays)
        self.assertIsNone(self.writer.histogram_num_buckets)


if __name__ == "__main__":
    absltest.main()
