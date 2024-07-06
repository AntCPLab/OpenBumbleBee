# Copyright 2024 Ant Group Co., Ltd.
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

__all__ = ["spu_silu"]

from functools import partial

from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def spu_silu(input):
    return _spu_silu_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _spu_silu_abstract(input):
    shape = input.shape
    dtype = dtypes.canonicalize_dtype(input.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _spu_silu_lowering(ctx, input):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input.type)

    call = custom_call(
        "spu.silu",
        # Output types
        result_types=[dtype],
        # The inputs:
        operands=[input],
    )

    return call.results


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _spu_silu_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some spu_silu of more complicated
# batching rules if you need such a thing.
def _spu_silu_batch(args, axes):
    assert axes[0] == axes[1]
    return spu_silu(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_spu_silu_prim = core.Primitive("spu_silu")
_spu_silu_prim.multiple_results = False
_spu_silu_prim.def_impl(partial(xla.apply_primitive, _spu_silu_prim))
_spu_silu_prim.def_abstract_eval(_spu_silu_abstract)

mlir.register_lowering(_spu_silu_prim, _spu_silu_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_spu_silu_prim] = _spu_silu_jvp
batching.primitive_batchers[_spu_silu_prim] = _spu_silu_batch
