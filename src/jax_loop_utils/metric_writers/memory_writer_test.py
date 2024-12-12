import chex
import jax
import jax.numpy as jnp
import pytest

from .memory_writer import (
    MemoryWriter,
    MemoryWriterAudioEntry,
    MemoryWriterHistogramEntry,
)


def test_write_scalars():
    writer = MemoryWriter()
    assert writer.scalars == {}
    writer.write_scalars(0, {"a": 3, "b": 0.15})
    assert writer.scalars == {0: {"a": 3, "b": 0.15}}
    writer.write_scalars(2, {"b": 0.007})
    assert writer.scalars == {0: {"a": 3, "b": 0.15}, 2: {"b": 0.007}}


def test_write_scalars_fails_when_using_same_step():
    writer = MemoryWriter()
    writer.write_scalars(0, {})
    with pytest.raises(
        ValueError, match=r"Step must be greater than the last inserted step\."
    ):
        writer.write_scalars(0, {})


def _image() -> jax.Array:
    return jnp.zeros((2, 2, 3), dtype=jnp.uint8)


def _video() -> jax.Array:
    return jnp.zeros((1, 5, 2, 2, 3), dtype=jnp.uint8)


def _audio() -> jax.Array:
    return jnp.zeros((8, 1), dtype=jnp.float16)


def _histogram() -> jax.Array:
    return jnp.zeros((10, 1))


def test_write_images():
    writer = MemoryWriter()
    assert writer.images == {}
    writer.write_images(1, {"a": _image()})
    chex.assert_trees_all_equal(dict(writer.images), {1: {"a": _image()}}, strict=True)


def test_write_videos():
    writer = MemoryWriter()
    assert writer.videos == {}
    writer.write_videos(2, {"a": _video()})
    chex.assert_trees_all_equal(dict(writer.videos), {2: {"a": _video()}}, strict=True)


def test_write_audios():
    writer = MemoryWriter()
    assert writer.audios == {}
    writer.write_audios(3, {"a": _audio()}, sample_rate=44100)
    chex.assert_trees_all_equal(
        dict(writer.audios),
        {3: MemoryWriterAudioEntry(audios={"a": _audio()}, sample_rate=44100)},
        strict=True,
    )


def test_write_texts():
    writer = MemoryWriter()
    assert writer.texts == {}
    writer.write_texts(4, {"a": "text"})
    assert writer.texts == {4: {"a": "text"}}


def test_write_histograms():
    writer = MemoryWriter()
    assert writer.histograms == {}
    writer.write_histograms(5, {"a": _histogram()})
    chex.assert_trees_all_equal(
        dict(writer.histograms),
        {5: MemoryWriterHistogramEntry(arrays={"a": _histogram()}, num_buckets=None)},
        strict=True,
    )
    writer.write_histograms(6, {"b": _histogram()}, num_buckets={"b": 10})
    chex.assert_trees_all_equal(
        dict(writer.histograms),
        {
            5: MemoryWriterHistogramEntry(arrays={"a": _histogram()}, num_buckets=None),
            6: MemoryWriterHistogramEntry(
                arrays={"b": _histogram()}, num_buckets={"b": 10}
            ),
        },
        strict=True,
    )


def test_write_hparams():
    writer = MemoryWriter()
    assert writer.hparams is None
    writer.write_hparams({"a": 3, "b": 0.15})
    assert writer.hparams == {"a": 3, "b": 0.15}


def test_write_hparams_fails_when_called_more_than_once():
    writer = MemoryWriter()
    writer.write_hparams({})
    with pytest.raises(AssertionError, match=r"Hyperparameters can only be set once\."):
        writer.write_hparams({})
