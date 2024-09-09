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

#include <optional>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetor::test {

class PrimeRingCastProtocolTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool>> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 18;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetor, PrimeRingCastProtocolTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false)),
    [](const testing::TestParamInfo<PrimeRingCastProtocolTest::ParamType> &p) {
      return fmt::format("{}Truc{}", std::get<0>(p.param),
                         (int)std::get<1>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<U>(x) >> (8 * sizeof(T) - 1) & 1;
}

void MaskItInplace(NdArrayRef a, size_t width) {
  auto field = a.eltype().as<Ring2k>()->field();
  if (width == SizeOf(field) * 8) {
    return;
  }
  DISPATCH_ALL_FIELDS(field, "mask", [&]() {
    NdArrayView<ring2k_t> _a(a);
    ring2k_t msk = (static_cast<ring2k_t>(1) << width) - 1;
    pforeach(0, _a.numel(), [&](int64_t i) { _a[i] &= msk; });
  });
}

// view [0, 2^k) as [-2^k/2, 2^k/2)
template <typename U>
auto ToSignType(U x, size_t width) {
  using S = typename std::make_signed<U>::type;
  if (sizeof(U) * 8 == width) {
    return static_cast<S>(x);
  }

  U half = static_cast<U>(1) << (width - 1);
  if (x >= half) {
    U upper = static_cast<U>(1) << width;
    x -= upper;
  }
  return static_cast<S>(x);
}

template <typename U>
auto ToSignTypePrime(U x, U prime) {
  using S = typename std::make_signed<U>::type;
  U half = prime >> 1;
  auto _x = static_cast<S>(x);
  return x <= half ? _x : _x - static_cast<S>(prime);
}

template <typename T>
T GetPrime();

template <>
uint32_t GetPrime() {
  return 2147377153ULL;
}

template <>
uint64_t GetPrime() {
  return 1152921504606584833ULL;
}

template <>
uint128_t GetPrime() {
  return static_cast<uint128_t>(GetPrime<uint64_t>()) * 1152921504606683137ULL;
}

template <typename sT>
sT idiv(sT a, sT n) {
  sT q = a / n;
  sT r = a % n;
  if (r != 0 and (std::signbit(r) != std::signbit(n))) {
    r += n;
    q -= 1;
  }
  return q;
}

template <typename T, typename sT>
sT TruncateByDef(T x, size_t shft, T modulus) {
  T upper = (modulus + 1) >> 1;
  sT sx = x;
  if (x >= upper) {
    sx -= modulus;
  }
  return idiv(sx, static_cast<sT>(1) << shft);
}

TEST_P(PrimeRingCastProtocolTest, Basic) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field = src_field;

  bool do_trunc = std::get<1>(GetParam());
  int shft = do_trunc ? 20 : 0;

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_zeros(src_field, shape);
  auto msg = ring_zeros(src_field, shape);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    using signedT = std::make_signed<ring2k_t>::type;
    signedT prime = GetPrime<ring2k_t>();

    std::uniform_int_distribution<signedT> uniform(
        static_cast<signedT>(1) -
            (prime >> PrimeRingCastProtocol::kHeuristicBound),
        (prime >> PrimeRingCastProtocol::kHeuristicBound) - 1);

    std::default_random_engine rdv;

    auto xmsg = NdArrayView<signedT>(msg);
    auto rnd0 = NdArrayView<ring2k_t>(inp[0]);
    auto rnd1 = NdArrayView<ring2k_t>(inp[1]);
    for (int64_t i = 0; i < shape.numel(); ++i) {
      rnd0[i] %= prime;

      xmsg[i] = uniform(rdv);

      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : prime - std::abs(xmsg[i]);
      rnd1[i] = (rnd1[i] + prime - rnd0[i]) % prime;
    }
  });

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    PrimeRingCastProtocol prot(base);
    PrimeRingCastProtocol::Meta meta;

    meta.dst_ring = dst_field;
    meta.dst_width = SizeOf(dst_field) * 8;
    meta.truncate_nbits = std::nullopt;
    if (do_trunc) {
      meta.truncate_nbits = shft;
    }

    DISPATCH_ALL_FIELDS(src_field, "set_prime", [&]() {
      meta.prime = ring_zeros(src_field, {1});
      meta.prime.at<ring2k_t>(0) = GetPrime<ring2k_t>();
      meta.prime_width = (int64_t)std::ceil(std::log2(GetPrime<ring2k_t>()));
    });

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_DEBUG("PrimeExt {} bits to {} bits sent {} bits per",
                 absl::bit_width(meta.prime), meta.dst_width,
                 (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<S0> expS(msg);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using U1 = ring2k_t;
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<U1> gotU(got);
      if (do_trunc) {
        size_t count0 = 0;
        size_t count1 = 0;
        size_t count2 = 0;
        for (int64_t i = 0; i < shape.numel(); ++i) {
          U1 ux = static_cast<U1>(expS[i] >> shft);
          if (ux == gotU[i]) {
            count0 += 1;
          } else if (ux < gotU[i]) {
            count1 += 1;
          } else {
            count2 += 1;
          }

          S1 diff = ux < gotU[i] ? (gotU[i] - ux) : ux - gotU[i];
          ASSERT_LE(diff, 1);
        }
        printf("0:%f, +1:%f, -1:%f\n", count0 * 1. / shape.numel(),
               count1 * 1. / shape.numel(), count2 * 1. / shape.numel());
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(static_cast<S1>(expS[i]), gotU[i]);
        }
      }
    });
  });
}

TEST_P(PrimeRingCastProtocolTest, RingUp) {
  size_t kWorldSize = 2;
  Shape shape = {kBenchSize};

  FieldType src_field = std::get<0>(GetParam());
  FieldType dst_field = FM128;
  if (src_field == FM32) {
    dst_field = FM64;
  }

  bool do_trunc = std::get<1>(GetParam());
  int shft = do_trunc ? 11 : 0;

  NdArrayRef inp[2];
  inp[0] = ring_rand(src_field, shape);
  inp[1] = ring_zeros(src_field, shape);
  auto msg = ring_zeros(src_field, shape);

  DISPATCH_ALL_FIELDS(src_field, "setup", [&]() {
    using signedT = std::make_signed<ring2k_t>::type;
    signedT prime = GetPrime<ring2k_t>();
    std::uniform_int_distribution<signedT> uniform(
        -(prime >> PrimeRingCastProtocol::kHeuristicBound),
        prime >> PrimeRingCastProtocol::kHeuristicBound);
    std::default_random_engine rdv;

    auto xmsg = NdArrayView<signedT>(msg);
    auto rnd0 = NdArrayView<ring2k_t>(inp[0]);
    auto rnd1 = NdArrayView<ring2k_t>(inp[1]);
    for (int64_t i = 0; i < shape.numel(); ++i) {
      rnd0[i] %= prime;

      xmsg[i] = uniform(rdv);

      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : prime - std::abs(xmsg[i]);
      rnd1[i] = (rnd1[i] + prime - rnd0[i]) % prime;
    }
  });

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);
    PrimeRingCastProtocol prot(base);
    PrimeRingCastProtocol::Meta meta;

    meta.dst_ring = dst_field;
    meta.dst_width = SizeOf(dst_field) * 8;
    meta.truncate_nbits = std::nullopt;
    if (do_trunc) {
      meta.truncate_nbits = shft;
    }

    DISPATCH_ALL_FIELDS(src_field, "set_prime", [&]() {
      meta.prime = ring_zeros(src_field, {1});
      meta.prime.at<ring2k_t>(0) = GetPrime<ring2k_t>();
      meta.prime_width = (int64_t)std::ceil(std::log2(GetPrime<ring2k_t>()));
    });

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    oup[rank] = prot.Compute(inp[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_DEBUG("PrimeExt {} bits to {} bits sent {} bits per",
                 absl::bit_width(meta.prime), meta.dst_width,
                 (b1 - b0) * 8. / shape.numel());
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(src_field, "check", [&]() {
    using S0 = std::make_signed<ring2k_t>::type;
    NdArrayView<S0> expS(msg);
    DISPATCH_ALL_FIELDS(dst_field, "check", [&]() {
      using S1 = std::make_signed<ring2k_t>::type;

      NdArrayView<S1> gotS(got);
      if (do_trunc) {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_NEAR(static_cast<S1>(expS[i]) >> shft, gotS[i], 1);
        }
      } else {
        for (int64_t i = 0; i < shape.numel(); ++i) {
          ASSERT_EQ(static_cast<S1>(expS[i]), gotS[i]);
        }
      }
    });
  });
}

}  // namespace spu::mpc::cheetor::test
