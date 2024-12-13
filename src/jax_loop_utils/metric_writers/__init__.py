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

"""Metric writers write ML model outputs during model training and evaluation.

This module introduces the MetricWriter interface. MetricWriters allow users
to write out metrics about ML models during training and evaluation (e.g. loss,
accuracy).
There is a MetricWriter implementation for each back end (e.g. TensorFlow
summaries) and classes that work on top other MetricWriter to
write to multiple writes at once or write asynchronously.

Note: The current interface might not contain write() methods for all possible
data types. We are open for extending the interface to other data types
(e.g. audio).

Usage:
  writer = MyMetricWriterImplementation()
  # Before training.
  writer.write_hparams({"learning_rate": 0.001, "batch_size": 64})
  # Start training loop.
  for step in range(num_train_steps):
    loss = train_step()
    if step % 50 == 0:
      writer.write_scalars(step, {"loss": loss})
      accuracy = evaluate()
      writer.write_scalars(step, {"accuracy": accuracy})
  # Make sure all values were written.
  writer.flush()  # or use metric_writers.ensure_flushes() context.
"""

from jax_loop_utils.metric_writers.async_writer import (
    AsyncMultiWriter,
    AsyncWriter,
    ensure_flushes,
)
from jax_loop_utils.metric_writers.interface import MetricWriter
from jax_loop_utils.metric_writers.keep_last_writer import KeepLastWriter
from jax_loop_utils.metric_writers.logging_writer import LoggingWriter
from jax_loop_utils.metric_writers.memory_writer import MemoryWriter
from jax_loop_utils.metric_writers.multi_writer import MultiWriter
from jax_loop_utils.metric_writers.prefix_suffix_writer import PrefixSuffixWriter
from jax_loop_utils.metric_writers.utils import write_values

# TODO(b/200953513): Migrate away from logging imports (on module level)
#                    to logging the actual usage. See b/200953513.

__all__ = [
    "AsyncMultiWriter",
    "AsyncWriter",
    "ensure_flushes",
    "KeepLastWriter",
    "LoggingWriter",
    "MemoryWriter",
    "MetricWriter",
    "MultiWriter",
    "PrefixSuffixWriter",
    "write_values",
]
