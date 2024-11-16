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

"""Interface for dataset iterators.

This module provides the DatasetIterator interface. This intention is that
several frameworks providing datasets can implement this interface without
knowing anything about the framework used for the model and the training loop.
Likewise can training loops assume to get an DatasetIterator object and do not
need to care about the specifics of the input pipelines.

This modules does not depend on TensorFlow. The interface is generic and users
don't have to use `tf.data` to construct a DatasetIterator. However, if they
use `tf.data` they can simply wrap their `tf.data.Dataset` object with
`TfDatasetIterator` to satisfy the interface.
"""
from __future__ import annotations

import abc
import collections.abc
import concurrent.futures
import dataclasses
import threading
from typing import Mapping, Optional, Sequence, Tuple, TypeVar, Union

from absl import logging
from jax_loop_utils import asynclib
from etils import epath
import jax.numpy as jnp  # Just for type checking.
import numpy as np
import numpy.typing as npt

Array = Union[np.ndarray, jnp.ndarray]
# Sizes of dimensions, None means the dimension size is unknown.
Shape = Tuple[Optional[int], ...]


@dataclasses.dataclass(frozen=True)
class ArraySpec:
  """Describes an array via it's dtype and shape."""
  dtype: npt.DTypeLike
  shape: Shape

  def __repr__(self):
    return f"ArraySpec(dtype={np.dtype(self.dtype).name}, shape={self.shape})"

  def __str__(self):
    return f"{np.dtype(self.dtype).name}{list(self.shape)}"


# Elements are PyTrees with NumPy/JAX arrays.

# Anything can be a PyTree (it's either a container or leaf). We define
# PyTree[T] as a PyTree where all leaves are of type T.
# See https://jax.readthedocs.io/en/latest/pytrees.html.
L = TypeVar("L")  # pylint: disable=invalid-name

PyTree = Union[L, Sequence["PyTree[L]"], Mapping[str, "PyTree[L]"]]

Element = PyTree[Array]
ElementSpec = PyTree[ArraySpec]


class DatasetIterator(collections.abc.Iterator):  # pytype: disable=ignored-abstractmethod
  """Generic interface for iterating over a dataset.

  This does not support __getitem__ since it cannot be implemented efficiently
  for many datasets. However datasets should allow starting the iterator from
  an arbitrary position.

  The element_spec property helps consumers to validate the input without
  reading data. This is similar to `tf.data.Dataset.element_spec`.

  Subclasses may decided to not read/write checkpoints if their state is
  sufficiently tracked externally (e.g. input pipelines that can be correctly
  restarted from the step number).
  """

  def get_next(self) -> Element:
    """Returns the next element."""
    logging.error(
        "DatasetIterator.get_next() is deprecated. Please use next().")
    # Subclasses should implement __next__() and remove calls to get_next().
    return next(self)

  def reset(self):
    """Resets the iterator back to the beginning."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def element_spec(self) -> ElementSpec:
    """Returns the spec elements."""
    raise NotImplementedError()

  def save(self, filename: epath.Path):
    """Saves the state of the iterator to a file.

    This should only handle this iterator - not iterators in other processes.

    Args:
      filename: Name of the checkpoint.
    """
    raise NotImplementedError()

  def restore(self, filename: epath.Path):
    """Restores the iterator from a file (if available).

    This should only handle this iterator - not iterators in other processes.

    Args:
      filename: Name of the checkpoint.
    """
    raise NotImplementedError()

  def load(self, filename: epath.Path):
    logging.error("DatasetIterator.load() is deprecated. Please use restore().")
    return self.restore(filename)





class PeekableDatasetIterator(DatasetIterator):
  """Wraps a DatasetIterator to provide a peek() method.

  This allows to look at the next element which can be useful in 2 scenarios:
  a) Get the structure of elements if the element_spec property is not
     supported.
  b) Request the next element without consuming it. This is especially handy to
     trigger reading of the first element while the model is being initialized.

  Example use case:
  >>> pool = clu.asynclib.Pool()
  >>> @pool
  >>> def warmup_input_pipeline():
  >>>   train_iter.peek()
  >>> first_batch_ready = warmup_input_pipeline()
  >>> # Do other stuff...
  >>> first_batch_ready.result()  # wait for input pipeline to be ready.
  """

  def __init__(self, it: DatasetIterator):
    self._it = it
    # Mutex for self._it.
    self._mutex = threading.Lock()
    self._peek: Optional[Element] = None
    self._pool = None
    self._peek_future = None

  def __next__(self) -> Element:
    with self._mutex:
      if self._peek is None:
        return next(self._it)
      peek = self._peek
      self._peek = None
      return peek

  def reset(self):
    with self._mutex:
      self._it.reset()
      self._peek = None
      self._pool = None
      self._peek_future = None

  @property
  def element_spec(self) -> ElementSpec:
    return self._it.element_spec

  def peek(self) -> Element:
    """Returns the next element without consuming it.

    This will get the next element from the underlying iterator. The element
    is stored and return on the next call of __next__().

    Returns:
      The next element.
    """
    if self._peek is None:
      self._peek = next(self)
    return self._peek

  def peek_async(self) -> concurrent.futures.Future[Element]:
    """Same as peek() but returns the Future of the element.

    Users can call this to warm up the iterator.

    Returns:
      Future with the next element. The element is also kept and returned on the
      next call of __next__().
    """
    with self._mutex:
      if self._peek_future is None:
        if self._pool is None:
          self._pool = asynclib.Pool(max_workers=1)
        self._peek_future = self._pool(self.peek)()
      return self._peek_future

  def save(self, filename: epath.Path):
    with self._mutex:
      self._it.save(filename)

  def restore(self, filename: epath.Path):
    with self._mutex:
      self._it.restore(filename)
