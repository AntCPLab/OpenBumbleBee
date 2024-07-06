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
from jax.lax import dot_general

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim


def batch_matmul():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = True

    sim = ppsim.Simulator(2, config)
    batch = 16
    x = (np.random.randn(batch, 64, 128) * 8.0).astype(int)
    y = (np.random.randn(batch, 128, 256) * 8.0).astype(int)
    target_func = lambda x, y: dot_general(x, y, ((2, 1), (0, 0)))

    spu_fn = ppsim.sim_jax(sim, target_func)
    z = spu_fn(x, y)
    g = target_func(x, y)
    diff = z - g

    print("batch matmul max diff = {}".format(np.max(diff)))


def matmul_with_interleave():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = True

    sim = ppsim.Simulator(2, config)

    # use int matrix to avoid truncation
    # To avoid truncation here because the baseOT might take too much time for init.
    x = (np.random.randn(128, 768) * 8.0).astype(int)
    y = (np.random.randn(768, 768) * 8.0).astype(int)

    spu_fn = ppsim.sim_jax(sim, jnp.dot)
    z = spu_fn(x, y)
    g = jnp.dot(x, y)
    diff = z - g

    print("matmul max diff = {}".format(np.max(diff)))


def matmul_with_packlwe():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False

    sim = ppsim.Simulator(2, config)

    # Dynamic packing: we turn to PackLWEs when the number of output is smaller than lattice dimension
    x = (np.random.randn(1, 50257) * 8.0).astype(int)
    y = (np.random.randn(50257, 768) * 8.0).astype(int)

    spu_fn = ppsim.sim_jax(sim, jnp.dot)
    z = spu_fn(x, y)
    g = jnp.dot(x, y)
    diff = z - g

    print("matmul max diff = {}".format(np.max(diff)))


if __name__ == "__main__":
    batch_matmul()
    # matmul_with_packlwe()
    # matmul_with_interleave()
