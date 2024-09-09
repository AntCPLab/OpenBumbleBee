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

#include "libspu/mpc/cheetah/dialect/cheetor/util.h"

#include <memory>

#include "seal/util/polyarithsmallmod.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace spu::mpc::cheetor {

namespace {

template <typename T>
T makeMask(size_t bw) {
  if (bw == sizeof(T) * 8) {
    return static_cast<T>(-1);
  }
  return (static_cast<T>(1) << bw) - 1;
}

template <typename T>
T makeTwoPow(size_t exp) {
  SPU_ENFORCE(exp < sizeof(T) * 8);
  return static_cast<T>(1) << exp;
}

}  // namespace

// Given x0 + x1 = x mod 2^k
// Compute h0 + h1 = x mod p with probability > 1 - |x|/2^k
void ProbConvRing2kShareToPrimeShare(const NdArrayRef& inp_share,
                                     absl::Span<uint64_t> oup_share,
                                     const seal::Modulus& prime, int rank) {
  SPU_ENFORCE(inp_share.eltype().isa<RingTy>());
  SPU_ENFORCE_EQ(inp_share.numel(), (int64_t)oup_share.size());
  SPU_ENFORCE(rank >= 0 && rank <= 1);

  auto eltype = inp_share.eltype();
  auto ring_ty = eltype.as<RingTy>()->field();
  size_t shr_width =
      eltype.isa<AShrTy>() ? eltype.as<AShrTy>()->nbits() : SizeOf(ring_ty) * 8;

  // x mod p
  DISPATCH_ALL_FIELDS(ring_ty, "ring2k_to_pime", [&]() {
    NdArrayView<const ring2k_t> inp(inp_share);
    auto msk = makeMask<ring2k_t>(shr_width);
    pforeach(0, oup_share.size(), [&](int64_t i) {
      // TODO(lwj): when shr_width is small, we can switch to faster
      // BarrettReduce.
      oup_share[i] = spu::mpc::cheetah::BarrettReduce(inp[i] & msk, prime);
    });
  });

  if (rank != 0) {
    return;
  }

  // Let rank=0 do the adjustments
  // h0 = (x0 - 2^k) mod P
  //    = (x0 mod P) + (-2^k mod P)
  uint64_t neg_2k = 1;
  if (shr_width < 8 * SizeOf(ring_ty)) {
    DISPATCH_ALL_FIELDS(ring_ty, "neg_2k", [&]() {
      neg_2k = spu::mpc::cheetah::BarrettReduce<ring2k_t>(
          makeTwoPow<ring2k_t>(shr_width), prime);
    });
  } else {
    SPU_ENFORCE((shr_width % 32) == 0);
    uint64_t _32k = static_cast<uint64_t>(1) << 32;
    auto _32 = seal::util::barrett_reduce_64(_32k, prime);
    // 2^64 mod p = (2^32 mod p)^2
    // 2^128 mod p = (2^32 mod p)^4
    for (size_t bw = 0; bw < shr_width; bw += 32) {
      // TODO(lwj): can use square to skip some mulmod
      neg_2k = seal::util::multiply_uint_mod(neg_2k, _32, prime);
    }
  }
  neg_2k = seal::util::negate_uint_mod(neg_2k, prime);

  seal::util::add_poly_scalar_coeffmod(oup_share.data(), oup_share.size(),
                                       neg_2k, prime, oup_share.data());
}

}  // namespace spu::mpc::cheetor
