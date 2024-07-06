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


def naive_softmax():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.fxp_exp_mode = 0
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(128, 32) * 8.0

    target_func = lambda x: jnn.softmax(x)
    spu_fn = ppsim.sim_jax(sim, target_func)

    z = spu_fn(x)
    g = target_func(x)

    diff = z - g
    print("max diff = {}".format(np.max(diff)))


def bumblebee_softmax():
    config = spu_pb2.RuntimeConfig(
        protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64
    )
    config.enable_hal_profile = True
    config.fxp_exp_mode = 0
    config.experimental_enable_colocated_optimization = False
    config.cheetah_2pc_config.enable_mul_lsb_error = True
    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    # enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
    copts.enable_optimize_denominator_with_broadcast = True

    def _softmax(x, axis=-1, where=None, initial=None):
        x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
        x = x - x_max
        # exp on large negative is clipped to zero
        nexp = si.spu_neg_exp(x)
        divisor = jnp.sum(nexp, axis, where=where, keepdims=True)
        return nexp / divisor

    sim = ppsim.Simulator(2, config)

    x = np.random.randn(128, 32) * 8.0

    target_func = jnn.softmax
    spu_fn = ppsim.sim_jax(sim, _softmax, copts=copts)

    z = spu_fn(x)
    g = target_func(x)

    diff = z - g
    print("max diff = {}".format(np.max(diff)))


if __name__ == "__main__":
    # naive_softmax()
    bumblebee_softmax()
