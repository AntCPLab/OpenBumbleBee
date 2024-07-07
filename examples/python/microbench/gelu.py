# Copyright 2023 Ant Group Co., Ltd.
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

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim


def gelu():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True
    config.cheetah_2pc_config.approx_less_precision = 4

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(1 << 10) * 4.0
    spu_fn = ppsim.sim_jax(sim, si.spu_gelu)
    z = spu_fn(x)
    g = jnn.gelu(x)
    diff = z - g

    # print(f"gelu spu out = {z[:10]}")
    # print(f"gelu cpu out = {g[:10]}")
    print("gelu max diff = {}".format(np.max(diff)))


def silu():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True
    config.cheetah_2pc_config.approx_less_precision = 4

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(1 << 20) * 8.0
    spu_fn = ppsim.sim_jax(sim, si.spu_silu)
    z = spu_fn(x)
    g = jnn.silu(x)
    diff = z - g

    # print(f"silu spu out = {z[:10]}")
    # print(f"silu cpu out = {g[:10]}")
    print("silu max diff = {}".format(np.max(diff)))


if __name__ == "__main__":
    gelu()
    # silu()
