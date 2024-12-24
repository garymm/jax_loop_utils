"""Utilities for audio and video.

Requires additional dependencies, part of the `audio-video` extra.
"""

import io

import av
import numpy as np

from jax_loop_utils.metric_writers.interface import (
    Array,
)

CONTAINER_FORMAT = "mp4"
CODEC = "h264"
FPS = 10


def encode_video(video_array: Array, destination: io.IOBase):
    """Encode a video array.

    Encodes using CODEC and writes using CONTAINER_FORMAT at FPS frames per second.

    Args:
        video_array: array to encode. Must have shape (T, H, W, 1) or (T, H, W, 3),
            where T is the number of frames, H is the height, W is the width, and the last
            dimension is the number of channels. Must have dtype uint8.
        destination: Destination to write the encoded video.
    """
    video_array = np.array(video_array)
    if (
        video_array.dtype != np.uint8
        or video_array.ndim != 4
        or video_array.shape[-1] not in (1, 3)
    ):
        raise ValueError(
            "Expected a uint8 array with shape (T, H, W, 1) or (T, H, W, 3)."
            f"Got shape {video_array.shape} with dtype {video_array.dtype}."
        )

    T, H, W, C = video_array.shape
    is_grayscale = C == 1
    if is_grayscale:
        video_array = np.squeeze(video_array, axis=-1)

    with av.open(destination, mode="w", format=CONTAINER_FORMAT) as container:
        stream = container.add_stream(CODEC, rate=FPS)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"

        for t in range(T):
            frame_data = video_array[t]
            if is_grayscale:
                # For grayscale, use gray format and let av handle conversion to yuv420p
                frame = av.VideoFrame.from_ndarray(frame_data, format="gray")
            else:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            frame.pts = t
            container.mux(stream.encode(frame))

        container.mux(stream.encode(None))
