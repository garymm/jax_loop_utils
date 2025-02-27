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


def _preprocess_video_array(video_array: Array) -> np.ndarray:
    video_array = np.array(video_array)
    if video_array.ndim != 4 or video_array.shape[-1] not in (1, 3):
        raise ValueError(
            "Expected an array with shape (T, H, W, 1) or (T, H, W, 3)."
            f"Got shape {video_array.shape} with dtype {video_array.dtype}."
        )

    if (
        np.issubdtype(video_array.dtype, np.floating)
        and np.all(video_array >= 0)
        and np.all(video_array <= 1.0)
    ):
        video_array = (video_array * 255).astype(np.uint8)
    elif (
        np.issubdtype(video_array.dtype, np.integer)
        and np.all(video_array >= 0)
        and np.all(video_array <= 255)
    ):
        video_array = video_array.astype(np.uint8)
    else:
        raise ValueError(
            "Expected video_array to be floats in [0, 1] "
            f"or ints in [0, 255], got {video_array.dtype}"
        )
    return video_array


def encode_video(
    video_array: Array,
    destination: io.IOBase,
    container_format: str = CONTAINER_FORMAT,
    codec: str = CODEC,
    fps: int = FPS,
):
    """Encode a video array.

    Encodes using CODEC and writes using CONTAINER_FORMAT at FPS frames per second.

    Args:
        video_array: array to encode. Must have shape (T, H, W, 1) or (T, H, W, 3),
            where T is the number of frames, H is the height, W is the width, and the last
            dimension is the number of channels.
            Must be ints in [0, 255] or floats in [0, 1].
        destination: Destination to write the encoded video.
    """
    video_array = _preprocess_video_array(video_array)

    T, H, W, C = video_array.shape
    # Pad height and width to even numbers if necessary
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        padding = [(0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        video_array = np.pad(video_array, padding, mode="constant")
        H += pad_h
        W += pad_w

    is_grayscale = C == 1
    if is_grayscale:
        video_array = np.squeeze(video_array, axis=-1)

    with av.open(destination, mode="w", format=container_format) as container:
        pix_fmt = "rgb8" if codec == "gif" else "yuv420p"
        stream = container.add_stream(codec, width=W, height=H, pix_fmt=pix_fmt, rate=fps)
        assert isinstance(stream, av.VideoStream)

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


def encode_video_to_gif(video_array: Array, destination: io.IOBase):
    """Encode a video array to a gif.

    Args:
        video_array: array to encode. Must have shape (T, H, W, 1) or (T, H, W, 3),
            where T is the number of frames, H is the height, W is the width, and the last
            dimension is the number of channels.
        destination: Destination to write the encoded video.
    """
    encode_video(video_array, destination, container_format="gif", codec="gif")
