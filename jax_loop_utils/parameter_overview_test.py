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

"""Tests for parameter overviews."""

from absl.testing import absltest
from jax_loop_utils import parameter_overview
import jax
import jax.numpy as jnp
import numpy as np


EMPTY_PARAMETER_OVERVIEW = """+------+-------+-------+------+------+-----+
| Name | Shape | Dtype | Size | Mean | Std |
+------+-------+-------+------+------+-----+
+------+-------+-------+------+------+-----+
Total: 0 -- 0 bytes"""

CONV2D_PARAMETER_OVERVIEW = """+-------------+--------------+---------+------+
| Name        | Shape        | Dtype   | Size |
+-------------+--------------+---------+------+
| conv/bias   | (2,)         | float32 | 2    |
| conv/kernel | (3, 3, 3, 2) | float32 | 54   |
+-------------+--------------+---------+------+
Total: 56 -- 224 bytes"""

CONV2D_PARAMETER_OVERVIEW_WITH_SHARDING = """+-------------+--------------+---------+------+----------+
| Name        | Shape        | Dtype   | Size | Sharding |
+-------------+--------------+---------+------+----------+
| conv/bias   | (2,)         | float32 | 2    | ()       |
| conv/kernel | (3, 3, 3, 2) | float32 | 54   | ()       |
+-------------+--------------+---------+------+----------+
Total: 56 -- 224 bytes"""

CONV2D_PARAMETER_OVERVIEW_WITH_STATS = """+-------------+--------------+---------+------+------+-----+
| Name        | Shape        | Dtype   | Size | Mean | Std |
+-------------+--------------+---------+------+------+-----+
| conv/bias   | (2,)         | float32 | 2    | 1.0  | 0.0 |
| conv/kernel | (3, 3, 3, 2) | float32 | 54   | 1.0  | 0.0 |
+-------------+--------------+---------+------+------+-----+
Total: 56 -- 224 bytes"""

CONV2D_PARAMETER_OVERVIEW_WITH_STATS_AND_SHARDING = """+-------------+--------------+---------+------+------+-----+----------+
| Name        | Shape        | Dtype   | Size | Mean | Std | Sharding |
+-------------+--------------+---------+------+------+-----+----------+
| conv/bias   | (2,)         | float32 | 2    | 1.0  | 0.0 | ()       |
| conv/kernel | (3, 3, 3, 2) | float32 | 54   | 1.0  | 0.0 | ()       |
+-------------+--------------+---------+------+------+-----+----------+
Total: 56 -- 224 bytes"""


class JaxParameterOverviewTest(absltest.TestCase):

  def test_count_parameters_empty(self):
    self.assertEqual(0, parameter_overview.count_parameters({}))

  def test_count_parameters(self):
    # Weights of a 2D convolution with 2 filters.
    params = {"conv": {"bias": jnp.ones((2,)), "kernel": jnp.ones((3, 3, 3, 2))}}
    # 3 * 3*3 * 2 + 2 (bias) = 56 parameters
    self.assertEqual(56,
                     parameter_overview.count_parameters(params))

  def test_get_parameter_overview_empty(self):
    self.assertEqual(EMPTY_PARAMETER_OVERVIEW,
                     parameter_overview.get_parameter_overview({}))

  def test_get_parameter_overview(self):
    # Weights of a 2D convolution with 2 filters.
    params = {"conv": {"bias": jnp.ones((2,)), "kernel": jnp.ones((3, 3, 3, 2))}}
    self.assertEqual(
        CONV2D_PARAMETER_OVERVIEW,
        parameter_overview.get_parameter_overview(params, include_stats=False))
    self.assertEqual(
        CONV2D_PARAMETER_OVERVIEW_WITH_STATS,
        parameter_overview.get_parameter_overview(params))
    # Add sharding with PartitionSpecs.
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), "d")
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params = jax.jit(lambda x: x, out_shardings=sharding)(params)
    self.assertEqual(
        CONV2D_PARAMETER_OVERVIEW_WITH_SHARDING,
        parameter_overview.get_parameter_overview(
            params, include_stats="sharding"))
    self.assertEqual(
        CONV2D_PARAMETER_OVERVIEW_WITH_STATS_AND_SHARDING,
        parameter_overview.get_parameter_overview(
            params, include_stats="global"))

  def test_get_parameter_overview_shape_dtype_struct(self):
    params = {"conv": {"bias": jnp.ones((2,)), "kernel": jnp.ones((3, 3, 3, 2))}}
    params_shape_dtype_struct = jax.eval_shape(lambda: params)
    self.assertEqual(
        CONV2D_PARAMETER_OVERVIEW,
        parameter_overview.get_parameter_overview(
            params_shape_dtype_struct, include_stats=False))

  def test_printing_bool(self):
    self.assertEqual(
        parameter_overview._default_table_value_formatter(True), "True")
    self.assertEqual(
        parameter_overview._default_table_value_formatter(False), "False")


if __name__ == "__main__":
  absltest.main()
