// Copyright 2024 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/dialect/cheetor/conversion.h"

#include <optional>

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/conversion.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/state.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/tiled_dispatch.h"
#include "libspu/mpc/common/pv2k.h"
namespace spu::mpc::cheetor {

NdArrayRef H2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  if (x.eltype().isa<AShrTy>()) {
    return x;
  }

  SPU_ENFORCE(x.eltype().isa<PrimeShrTy>());
  auto ft = x.eltype().as<RingTy>()->field();
  return cheetah::DispatchUnaryFunc(
             ctx, x,
             [&](const NdArrayRef& input,
                 const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
               PrimeRingCastProtocol::Meta meta;
               meta.dst_ring = ft;
               meta.dst_width = x.eltype().as<PrimeShrTy>()->nbits();
               meta.truncate_nbits = std::nullopt;  //  no truncation
               meta.prime = CheetorMulProt::GetPrimesProduct(ft);
               PrimeRingCastProtocol prot(base);
               return prot.Compute(input, meta);
             })
      .as(makeType<AShrTy>(ft, x.eltype().as<PrimeShrTy>()->nbits()));
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  auto ft = in.eltype().as<RingTy>()->field();
  return cheetah::B2A{}.proc(ctx, in).as(makeType<AShrTy>(ft));
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  if (in.eltype().isa<AShrTy>()) {
    const auto field = in.eltype().as<Ring2k>()->field();
    auto* comm = ctx->getState<Communicator>();
    auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
    return out.as(makeType<Pub2kTy>(field));
  }

  if (in.eltype().isa<PrimeShrTy>()) {
    // call H2A first
    return proc(ctx, H2A{}.proc(ctx, in));
  }

  SPU_THROW("invalid eltype={}", in.eltype());
}

}  // namespace spu::mpc::cheetor
