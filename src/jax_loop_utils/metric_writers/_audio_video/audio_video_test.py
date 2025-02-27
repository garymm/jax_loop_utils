"""Tests for video encoding utilities."""

import io

import av
import numpy as np
from absl.testing import absltest

from jax_loop_utils.metric_writers._audio_video import (
    CONTAINER_FORMAT,
    FPS,
    encode_video,
    encode_video_to_gif,
)


class VideoTest(absltest.TestCase):
    """Tests for video encoding utilities."""

    def test_encode_video_invalid_args(self):
        """Test that encode_video raises appropriate errors for invalid inputs."""
        invalid_shape = np.zeros((10, 20, 30, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, r"Expected an array with shape"):
            encode_video(invalid_shape, io.BytesIO())

        invalid_dtype = 2 * np.ones((10, 20, 30, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"Expected video_array to be floats in \[0, 1\]"):
            encode_video(invalid_dtype, io.BytesIO())

    def test_encode_video_success(self):
        # Create a simple test video - red square moving diagonally
        T, H, W = 20, 63, 63  # test non-even dimensions
        video = np.zeros((T, H, W, 3), dtype=np.uint8)
        for t in range(T):
            pos = t * 5  # Move 5 pixels each frame
            video[t, pos : pos + 10, pos : pos + 10, 0] = 255  # Red square

        output = io.BytesIO()
        encode_video(video, output)

        output.seek(0)
        with av.open(output, mode="r", format=CONTAINER_FORMAT) as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.width, W + 1)
            self.assertEqual(stream.codec_context.height, H + 1)
            self.assertEqual(stream.codec_context.framerate, FPS)
            # Check we can decode all frames
            frame_count = sum(1 for _ in container.decode(stream))
            self.assertEqual(frame_count, T)

    def test_encode_video_to_gif(self):
        # Create a simple test video - red square moving diagonally
        T, H, W = 20, 63, 63  # test non-even dimensions
        video = np.zeros((T, H, W, 3), dtype=np.uint8)
        for t in range(T):
            pos = t * 5  # Move 5 pixels each frame
            video[t, pos : pos + 10, pos : pos + 10, 0] = 255  # Red square

        output = io.BytesIO()
        encode_video_to_gif(video, output)

        output.seek(0)
        with av.open(output, mode="r", format="gif") as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.width, W + 1)
            self.assertEqual(stream.codec_context.height, H + 1)
            # Check we can decode all frames
            frame_count = sum(1 for _ in container.decode(stream))
            self.assertEqual(frame_count, T)

    def test_encode_video_grayscale(self):
        T, H, W = 5, 32, 32
        video = np.zeros((T, H, W, 1), dtype=np.uint8)

        # Create pulsing pattern
        for t in range(T):
            video[t, :, :, 0] = (t * 50) % 256  # Increasing brightness

        output = io.BytesIO()
        encode_video(video, output)

        output.seek(0)
        with av.open(output, mode="r", format=CONTAINER_FORMAT) as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.width, W)
            self.assertEqual(stream.codec_context.height, H)
            # Check we can decode all frames
            frame_count = sum(1 for _ in container.decode(stream))
            self.assertEqual(frame_count, T)


if __name__ == "__main__":
    absltest.main()
