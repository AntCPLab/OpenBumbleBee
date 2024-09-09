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

#include "libspu/mpc/cheetah/dialect/cheetor/arithmetic.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/state.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/tiled_dispatch.h"
#include "libspu/mpc/cheetah/truncate.h"
namespace spu::mpc::cheetor {

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.eltype(), y.eltype());
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  SPU_ENFORCE(x.eltype().isa<AShrTy>() or x.eltype().isa<cheetah::AShrTy>());

  if (x.numel() == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  if (x.data() == y.data() && x.strides() == y.strides() &&
      x.offset() == y.offset()) {
    return SquareA{}.proc(ctx, x);
  }

  auto mul_prot = ctx->getState<CheetorMulState>()->GetMulProt();
  FieldType ft = x.eltype().as<RingTy>()->field();
  mul_prot->LazyInit(ft);

  return mul_prot->RingShareToPrimeShareMul(x, y);
}

NdArrayRef SquareA::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  SPU_ENFORCE(x.eltype().isa<AShrTy>() or x.eltype().isa<cheetah::AShrTy>());
  FieldType ft = x.eltype().as<RingTy>()->field();

  if (x.numel() == 0) {
    return NdArrayRef(makeType<PrimeShrTy>(ft), x.shape());
  }

  auto mul_prot = ctx->getState<CheetorMulState>()->GetMulProt();
  mul_prot->LazyInit(ft);

  return mul_prot->RingShareToPrimeShareSquare(x).as(makeType<PrimeShrTy>(ft));
}

NdArrayRef TruncPrA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          size_t bits, SignType sign) const {
  SPU_ENFORCE(x.eltype().isa<PrimeShrTy>());

  auto ft = x.eltype().as<RingTy>()->field();
  size_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(makeType<AShrTy>(ft), x.shape());
  }

  if (x.eltype().isa<AShrTy>() or x.eltype().isa<cheetah::AShrTy>()) {
    // Truncate on AShr switch to cheetah's TruncPrA.
    cheetah::TruncPrA trunc_pr;
    return trunc_pr.proc(ctx, x, bits, sign).as(x.eltype());
  }

  // For PrimeShr, we perform the prime-to-ring conversion and
  // truncation together.
  return cheetah::DispatchUnaryFunc(
             ctx, x,
             [&](const NdArrayRef& input,
                 const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
               PrimeRingCastProtocol::Meta meta;
               meta.dst_ring = ft;
               meta.dst_width = x.eltype().as<PrimeShrTy>()->nbits();
               int trunc_nbits = static_cast<int>(bits);
               meta.truncate_nbits = trunc_nbits;
               meta.prime = CheetorMulProt::GetPrimesProduct(ft);
               meta.prime_width = CheetorMulProt::PrimeNumBits(ft);

               PrimeRingCastProtocol prot(base);
               return prot.Compute(input, meta);
             })
      .as(makeType<AShrTy>(ft, x.eltype().as<PrimeShrTy>()->nbits()));
}

NdArrayRef TruncPrimeToRing::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                                  size_t trunc_bits, FieldType target_ring,
                                  size_t target_ring_width) const {
  SPU_ENFORCE(x.eltype().isa<PrimeShrTy>());
  SPU_ENFORCE(target_ring_width > 0 and
              target_ring_width <= SizeOf(target_ring) * 8);

  FieldType ft = x.eltype().as<PrimeShrTy>()->field();
  SPU_ENFORCE(target_ring_width > CheetorMulProt::PrimeNumBits(ft));

  return cheetah::DispatchUnaryFunc(
             ctx, x,
             [&](const NdArrayRef& input,
                 const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
               PrimeRingCastProtocol::Meta meta;
               meta.dst_ring = ft;
               meta.dst_width = x.eltype().as<PrimeShrTy>()->nbits();
               int trunc_nbits = static_cast<int>(trunc_bits);
               meta.truncate_nbits = trunc_nbits;
               meta.prime = CheetorMulProt::GetPrimesProduct(ft);
               meta.prime_width = CheetorMulProt::PrimeNumBits(ft);

               PrimeRingCastProtocol prot(base);
               return prot.Compute(input, meta);
             })
      .as(makeType<AShrTy>(ft, target_ring_width));
}

// Given x in [0, 2^k)
// Compute x^2, x^3, x^4 mod 2^k
// The multiplications are done in the specified prime field.
// The final results are converted to modulus 2^k.
// std::array<NdArrayRef, 3> Power4::proc(KernelEvalContext* ctx,
//                                        const NdArrayRef& x,
//                                        FieldType target_field,
//                                        bool is_fxp) const {}
}  // namespace spu::mpc::cheetor
