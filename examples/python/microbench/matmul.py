# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim


def matmul_with_interleave():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.BUMBLEBEE, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False

    sim = ppsim.Simulator(2, config)

    # use int matrix to avoid truncation
    # To avoid truncation here because the baseOT might take too much time for init.
    x = (np.random.randn(128, 128) * 8.0).astype(int)
    y = (np.random.randn(128, 1024) * 8.0).astype(int)

    spu_fn = ppsim.sim_jax(sim, jnp.dot)
    z = spu_fn(x, y)
    g = jnp.dot(x, y)
    diff = z - g

    print("matmul max diff = {}".format(np.max(diff)))


def matmul_with_packlwe():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.BUMBLEBEE, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False

    sim = ppsim.Simulator(2, config)

    # Dynamic packing: we turn to PackLWEs when the number of output is smaller than lattice dimension
    x = (np.random.randn(16, 256) * 8.0).astype(int)
    y = (np.random.randn(256, 8) * 8.0).astype(int)

    spu_fn = ppsim.sim_jax(sim, jnp.dot)
    z = spu_fn(x, y)
    g = jnp.dot(x, y)
    diff = z - g

    print("matmul max diff = {}".format(np.max(diff)))


if __name__ == "__main__":
    matmul_with_packlwe()
    matmul_with_interleave()
