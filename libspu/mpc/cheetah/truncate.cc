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

#include "libspu/mpc/cheetah/truncate.h"

#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/tiled_dispatch.h"
namespace spu::mpc::cheetah {

NdArrayRef TruncPrA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          size_t bits, SignType sign) const {
  size_t n = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (n == 0) {
    return out;
  }

  if (x.eltype().isa<cheetor::PrimeShrTy>()) {
    // For a prime share (i.e., from cheetor::mul),
    // we perform the prime-to-ring conversion with free-truncation.
    // NOTE(lwj): for lazy truncation such as
    //   _trunc(_add(_mul(a, b), _mul(c, d)))
    // The 'add' is still carried out over the ring.
    // Thus there might be a possibility (if lazy too much) of overflowing 2^k
    // and ruins the prime share.
    auto ft = x.eltype().as<RingTy>()->field();
    return DispatchUnaryFunc(
               ctx, x,
               [&](const NdArrayRef& input,
                   const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
                 cheetor::PrimeRingCastProtocol::Meta meta;
                 meta.dst_ring = ft;
                 meta.dst_width = x.eltype().as<cheetor::PrimeShrTy>()->nbits();
                 int trunc_nbits = static_cast<int>(bits);
                 meta.truncate_nbits = trunc_nbits;
                 meta.prime = cheetor::CheetorMulProt::GetPrimesProduct(ft);
                 meta.prime_width = cheetor::CheetorMulProt::PrimeNumBits(ft);

                 cheetor::PrimeRingCastProtocol prot(base);
                 return prot.Compute(input, meta);
               })
        .as(makeType<AShrTy>(ft));
  } else {
    // normal ring share truncation
    return DispatchUnaryFunc(
        ctx, x,
        [&](const NdArrayRef& input,
            const std::shared_ptr<BasicOTProtocols>& base_ot) {
          TruncateProtocol::Meta meta;
          meta.signed_arith = true;
          meta.sign = sign;
          meta.shift_bits = bits;
          meta.use_heuristic = true;
          TruncateProtocol prot(base_ot);
          return prot.Compute(input, meta);
        });
  }
}
}  // namespace spu::mpc::cheetah
