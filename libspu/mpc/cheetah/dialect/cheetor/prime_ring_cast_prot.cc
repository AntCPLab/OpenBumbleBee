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

#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetor {

namespace {
template <typename T>
auto makeMask(size_t bw) {
  using U = typename std::make_unsigned<T>::type;
  if (bw == sizeof(U) * 8) {
    return static_cast<U>(-1);
  }
  return (static_cast<U>(1) << bw) - 1;
}

[[maybe_unused]] NdArrayRef CastTo(const NdArrayRef &x, FieldType from,
                                   FieldType to) {
  SPU_ENFORCE_EQ(x.eltype().as<RingTy>()->field(), from);
  // cast from src type to dst type
  auto out = ring_zeros(to, x.shape());
  DISPATCH_ALL_FIELDS(from, "cast_to", [&]() {
    using U0 = ring2k_t;
    NdArrayView<U0> src(x);

    DISPATCH_ALL_FIELDS(to, "cast_to", [&]() {
      using U1 = ring2k_t;
      NdArrayView<U1> dst(out);
      pforeach(0, src.numel(),
               [&](int64_t i) { dst[i] = static_cast<U1>(src[i]); });
    });
  });
  return out;
}

}  // namespace

PrimeRingCastProtocol::PrimeRingCastProtocol(
    const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> &base)
    : basic_ot_prot_(base) {}

// We assume the input is ``positive''
// Given h0 + h1 = h mod p and h < p / 2
// Define b0 = 1{h0 >= p/2}
//        b1 = 1{h1 >= p/2}
// Compute w = 1{h0 + h1 >= p}
NdArrayRef PrimeRingCastProtocol::MSB0ToWrap(const NdArrayRef &inp,
                                             const Meta &meta) {
  const auto src_ring = inp.eltype().as<Ring2k>()->field();
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t bw = meta.dst_width;

  NdArrayRef cot_output = ring_zeros(meta.dst_ring, inp.shape());
  DISPATCH_ALL_FIELDS(src_ring, "MSB1ToWrap", [&]() {
    using U0 = std::make_unsigned<ring2k_t>::type;
    DISPATCH_ALL_FIELDS(meta.dst_ring, "MSB1ToWrap", [&]() {
      using U1 = std::make_unsigned<ring2k_t>::type;

      NdArrayView<const U0> xinp(inp);
      auto xout = absl::MakeSpan(&cot_output.at<U1>(0), cot_output.numel());
      auto phalf = (meta.prime.at<U0>(0) + 1) >> 1;
      auto msk = makeMask<U1>(meta.dst_width);

      if (rank == 0) {
        std::vector<U1> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = 1 - static_cast<U1>(xinp[i] >= phalf);
        });

        auto sender = basic_ot_prot_->GetSenderCOT();
        sender->SendCAMCC(absl::MakeConstSpan(cot_input), xout, bw);
        sender->Flush();

      } else {
        std::vector<uint8_t> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = 1 - static_cast<uint8_t>(xinp[i] >= phalf);
        });

        basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(
            absl::MakeConstSpan(cot_input), xout, bw);

        pforeach(0, numel, [&](int64_t i) { xout[i] = (1 - xout[i]) & msk; });
      }
    });
  });

  return cot_output.as(makeType<spu::mpc::cheetah::BShrTy>(meta.dst_ring, 1));
}

NdArrayRef PrimeRingCastProtocol::Compute(const NdArrayRef &inp,
                                          const Meta &meta) {
  const auto src_ring = inp.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(meta.prime.numel() == 1 and meta.prime.elsize() == inp.elsize());
  SPU_ENFORCE(meta.prime_width >= 1 and meta.prime_width <= 128,
              "meta.prime_width={}", meta.prime_width);
  size_t prime_width = meta.prime_width;
  SPU_ENFORCE(SizeOf(src_ring) * 8 >= prime_width);
  SPU_ENFORCE(meta.dst_width > (int64_t)prime_width);
  SPU_ENFORCE(meta.dst_width <= (int64_t)SizeOf(meta.dst_ring) * 8);

  auto truncate_nbits = meta.truncate_nbits ? *meta.truncate_nbits : 0;
  SPU_ENFORCE(truncate_nbits >= 0, "invalid truncate_nbits={}", truncate_nbits);

  DISPATCH_ALL_FIELDS(src_ring, "check_range", [&]() {
    NdArrayView<const ring2k_t> input(inp);
    ring2k_t prime = meta.prime.at<ring2k_t>(0);
    for (int64_t i = 0; i < input.numel(); ++i) {
      SPU_ENFORCE(input[i] < prime, "prime share out-of-range");
    }
  });

  const int rank = basic_ot_prot_->Rank();
  const int shft = truncate_nbits;
  NdArrayRef outp = ring_zeros(meta.dst_ring, inp.shape());
  if (rank == 0) {
    auto wrap_arith = MSB0ToWrap(inp, meta);
    DISPATCH_ALL_FIELDS(src_ring, "finalize", [&]() {
      using U0 = ring2k_t;
      U0 prime = meta.prime.at<U0>(0);
      NdArrayView<const U0> input(inp);
      DISPATCH_ALL_FIELDS(meta.dst_ring, "finalize", [&]() {
        using U1 = ring2k_t;
        auto msk = makeMask<U1>(meta.dst_width);
        NdArrayView<U1> output(outp);
        NdArrayView<const U1> wrap(wrap_arith);
        pforeach(0, inp.numel(), [&](int64_t i) {
          output[i] = static_cast<U1>(input[i] >> shft) -
                      static_cast<U1>(prime >> shft) * wrap[i];
          output[i] &= msk;
        });
      });
    });

    return outp.as(makeType<AShrTy>(meta.dst_ring, meta.dst_width));
  }

  /// rank = 1
  auto adjusted = inp.clone();
  DISPATCH_ALL_FIELDS(src_ring, "wrap.adj", [&]() {
    using U0 = ring2k_t;
    NdArrayView<U0> xadj(adjusted);
    U0 prime = meta.prime.at<U0>(0);
    U0 big_val = static_cast<U0>(1) << (prime_width - kHeuristicBound);

    // add a big value (then modulo prime) so that the prime share
    // shoud be positive now.
    pforeach(0, xadj.numel(), [&](int64_t i) {
      xadj[i] = big_val + xadj[i];
      xadj[i] -= (xadj[i] >= prime ? prime : 0);
    });
  });

  // Wrap w = 1{h0 + h1 >= p}
  auto wrap_arith = MSB0ToWrap(adjusted, meta);

  DISPATCH_ALL_FIELDS(src_ring, "finalize", [&]() {
    using U0 = ring2k_t;
    U0 prime = meta.prime.at<U0>(0);
    NdArrayView<const U0> input(adjusted);
    DISPATCH_ALL_FIELDS(meta.dst_ring, "finalize", [&]() {
      using U1 = ring2k_t;
      const auto msk = makeMask<U1>(meta.dst_width);
      U1 big_val = static_cast<U1>(1) << (prime_width - kHeuristicBound - shft);

      NdArrayView<U1> output(outp);
      NdArrayView<const U1> wrap(wrap_arith);
      pforeach(0, inp.numel(), [&](int64_t i) {
        // Result is (h1/2^d) - (p/2^d) * w1 mod 2^k.
        output[i] = static_cast<U1>(input[i] >> shft) -
                    static_cast<U1>(prime >> shft) * wrap[i];
        // Remove the (shifted) big val from the mod-2^k share.
        output[i] = (output[i] - big_val) & msk;
      });
    });
  });

  return outp.as(makeType<AShrTy>(meta.dst_ring, meta.dst_width));
}

}  // namespace spu::mpc::cheetor
