import os

from jax_loop_utils.data import ArraySpec,DatasetIterator, Element, ElementSpec
from etils import epath
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise RuntimeError("tf_dataset_iterator requires tensorflow.")

class TfDatasetIterator(DatasetIterator):
  """DatasetIterator for wrapping a `tf.data.Dataset`."""

  def __init__(self, dataset, *, checkpoint: bool):
    """Wraps `tf.data.Dataset` object into the `DatasetIterator` interface.

    Warning: Do not wrap this interator to do asynchronous prefetching if you
    use `checkpoint=True` (default). tf.data iterators must be saved()
    synchronously.

    Args:
      dataset: The dataset to wrap. Elements are converted to NumPy arrays but
        no additional prefetching is done. tf.data should automatically prefetch
        elements (to CPU memory).
      checkpoint: Whether to checkpoint the dataset iterator object.
        Checkpointing dataset iterators is required for handling job
        pre-emptions but depending on your input pipeline can result in very
        large checkpoints. If set to False save() and load() are no-ops.
    """

    if not isinstance(dataset, tf.data.Dataset):
      raise ValueError("`dataset` must be an instance of `tf.data.Dataset` "
                       f"but got {type(dataset)}.")
    self._dataset = dataset
    self._checkpoint = checkpoint
    assert self.element_spec  # Verify element spec.
    self.iterator = iter(dataset)
    self._ckpt = tf.train.Checkpoint(ds=self.iterator)

  def get_next(self) -> Element:
    return next(self)

  def __next__(self) -> Element:
    return {k: np.asarray(v) for k, v in next(self.iterator).items()}

  def reset(self):
    self.iterator = iter(self._dataset)
    self._ckpt = tf.train.Checkpoint(ds=self.iterator)

  @property
  def element_spec(self) -> ElementSpec:
    element_spec = self._dataset.element_spec
    if not isinstance(element_spec, dict):
      raise ValueError("Dataset elements must be flat dictionaries but got "
                       f"{element_spec}.")
    invalid_features = [
        k for k, v in element_spec.items()
        if not isinstance(v, tf.TensorSpec)
    ]
    if invalid_features:
      raise ValueError(f"Features {invalid_features} are not tensors. Dataset "
                       "elements must be flat dictionaries of tensors.")
    return {
        k: ArraySpec(dtype=v.dtype.as_numpy_dtype, shape=tuple(v.shape))
        for k, v in element_spec.items()
    }

  def save(self, filename: epath.Path):
    if self._checkpoint:
      self._ckpt.write(os.fspath(filename))

  def restore(self, filename: epath.Path):
    if self._checkpoint:
      self._ckpt.read(os.fspath(filename)).assert_consumed()
