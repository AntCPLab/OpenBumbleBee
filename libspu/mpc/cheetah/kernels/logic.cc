// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/cheetah/kernels/logic.h"

#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {
NdArrayRef makeBShare(const NdArrayRef& r, FieldType field, size_t nbits) {
  const auto ty = makeType<BShrTy>(field, nbits);
  return r.as(ty);
}
}  // namespace

NdArrayRef LShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t out_nbits = in.eltype().as<BShare>()->nbits() + shift;
  out_nbits = std::clamp(out_nbits, static_cast<size_t>(0), SizeOf(field) * 8);

  return makeBShare(ring_lshift(in, shift), field, out_nbits);
}

NdArrayRef RShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  size_t nbits = in.eltype().as<BShare>()->nbits();
  size_t out_nbits = nbits - std::min(nbits, shift);
  SPU_ENFORCE(nbits <= SizeOf(field) * 8);

  return makeBShare(ring_rshift(in, shift), field, out_nbits);
}

NdArrayRef ARShiftB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t shift) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  shift %= SizeOf(field) * 8;

  // arithmetic right shift expects to work on ring, or the behaviour is
  // undefined.
  return makeBShare(ring_arshift(in, shift), field, SizeOf(field) * 8);
}

NdArrayRef NotA::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto* comm = ctx->getState<Communicator>();
  auto res = ring_neg(in);
  if (comm->getRank() == 0) {
    const auto field = in.eltype().as<Ring2k>()->field();
    ring_add_(res, ring_not(ring_zeros(field, in.shape())));
  }

  return res.as(in.eltype());
}

NdArrayRef LShiftA::proc(KernelEvalContext*, const NdArrayRef& in,
                         size_t bits) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;

  return ring_lshift(in, bits).as(in.eltype());
}
}  // namespace spu::mpc::cheetah
