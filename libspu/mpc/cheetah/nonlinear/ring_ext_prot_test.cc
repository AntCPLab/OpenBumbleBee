// Copyright 2022 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"

#include <random>

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/rlwe.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class RingExtProtTest : public ::testing::TestWithParam<
                            std::tuple<FieldType, bool, std::string>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, RingExtProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(true, false),
                     testing::Values("Unknown", "Zero", "One")),
    [](const testing::TestParamInfo<RingExtProtTest::ParamType> &p) {
      return fmt::format("{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using uT = typename std::make_unsigned<T>::type;
  return (static_cast<uT>(x) >> (8 * sizeof(T) - 1)) & 1;
}

TEST_P(RingExtProtTest, Basic) {
  size_t kWorldSize = 2;
  int64_t n = 1 << 10;

  FieldType field = std::get<0>(GetParam());
  FieldType dst_field;
  if (field == FieldType::FM32) {
    dst_field = FieldType::FM64;
  } else {
    dst_field = FieldType::FM128;
  }
  size_t src_ring_width = SizeOf(field) * 8;

  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());

  SignType sign;
  NdArrayRef inp[2];
  inp[0] = ring_rand(field, {n});

  if (msb == "Unknown") {
    inp[1] = ring_rand(field, {n});
    sign = SignType::Unknown;
  } else {
    auto msg = ring_rand(field, {n});
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      auto xmsg = NdArrayView<ring2k_t>(msg);
      size_t bw = SizeOf(field) * 8;
      if (msb == "Zero") {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] &= mask; });
        sign = SignType::Positive;
      } else {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1));
        pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] |= mask; });
        sign = SignType::Negative;
      }
    });

    inp[1] = ring_sub(msg, inp[0]);
  }

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtProtocol prot(base);
    RingExtProtocol::Meta meta;

    meta.sign = sign;
    meta.signed_arith = signed_arith;
    meta.src_field = field;
    meta.src_ring_width = src_ring_width;
    meta.dst_field = dst_field;
    meta.dst_ring_width = SizeOf(dst_field) * 8;
    meta.use_heuristic = false;

    [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();

    oup[rank] = prot.Compute(inp[rank], meta);

    [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

    SPDLOG_DEBUG("Ext {} bits {} elements sent {} bytes, {} bits each #sent {}",
                 meta.src_ring_width, inp[0].numel(), (b1 - b0),
                 (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
  });

  EXPECT_EQ(oup[0].shape(), oup[1].shape());
  auto expected = ring_add(inp[0], inp[1]);
  auto got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;
    using sT0 = std::make_signed<ring2k_t>::type;
    NdArrayView<const T0> _exp(expected);
    NdArrayView<const sT0> _signed_exp(expected);
    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      using sT1 = std::make_signed<ring2k_t>::type;
      NdArrayView<const T1> _got(got);
      NdArrayView<const sT1> _signed_got(got);

      if (signed_arith) {
        for (int64_t i = 0; i < _got.numel(); ++i) {
          EXPECT_EQ(static_cast<sT1>(_signed_exp[i]), _signed_got[i]);
        }
      } else {
        for (int64_t i = 0; i < _got.numel(); ++i) {
          ASSERT_EQ(static_cast<T1>(_exp[i]), _got[i]);
        }
      }
    });
  });
}

TEST_P(RingExtProtTest, Heuristic) {
  size_t kWorldSize = 2;
  int64_t n = 1 << 19;

  FieldType field = std::get<0>(GetParam());
  FieldType dst_field;
  if (field == FieldType::FM32) {
    dst_field = FieldType::FM64;
  } else {
    dst_field = FieldType::FM128;
  }

  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());

  if (not signed_arith || msb != "Unknown") {
    return;
  }

  NdArrayRef input[2];
  input[0] = ring_rand(field, {n});

  SignType sign = SignType::Unknown;

  NdArrayRef msg;
  int kHeuristicBound = 4;
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    // message is small enough
    auto msg = ring_rand(field, {n});
    ring_rshift_(msg, kHeuristicBound);

    NdArrayView<ring2k_t> xmsg(msg);
    // some message are negative
    for (int64_t i = 0; i < n; i += 2) {
      xmsg[i] = -xmsg[i];
    }
    input[1] = ring_sub(msg, input[0]);
  });

  NdArrayRef output[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    RingExtProtocol ext_prot(base);
    RingExtProtocol ::Meta meta;

    meta.sign = sign;
    meta.signed_arith = signed_arith;
    meta.src_field = field;
    meta.dst_field = dst_field;
    meta.dst_ring_width = SizeOf(dst_field) * 8;
    meta.use_heuristic = true;

    (void)ext_prot.Compute(input[rank].slice({0}, {1}, {1}), meta);

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] size_t s0 = ctx->GetStats()->sent_actions.load();

    output[rank] = ext_prot.Compute(input[rank], meta);

    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes.load();
    [[maybe_unused]] size_t s1 = ctx->GetStats()->sent_actions.load();

    SPDLOG_INFO("Ext {} bits {} elements sent {} bytes, {} bits each #sent {}",
                meta.src_ring_width, n, (b1 - b0), (b1 - b0) * 8. / n,
                (s1 - s0));
  });

  auto expected = ring_add(input[0], input[1]);
  auto got = ring_add(output[0], output[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;
    using sT0 = std::make_signed<ring2k_t>::type;
    NdArrayView<const T0> _exp(expected);
    NdArrayView<const sT0> _signed_exp(expected);
    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      using sT1 = std::make_signed<ring2k_t>::type;
      NdArrayView<const T1> _got(got);
      NdArrayView<const sT1> _signed_got(got);

      for (int64_t i = 0; i < _got.numel(); ++i) {
        EXPECT_EQ(static_cast<sT1>(_signed_exp[i]), _signed_got[i]);
      }
    });
  });
}

template <typename T>
T GetPrime();

template <>
uint32_t GetPrime() {
  return (static_cast<uint32_t>(1) << 31) - 1;
}

template <>
uint64_t GetPrime() {
  return (static_cast<uint64_t>(1) << 61) - 1;
}

template <>
uint128_t GetPrime() {
  return (static_cast<uint128_t>(1) << 113) - 1;
}

TEST_P(RingExtProtTest, Prime) {
  size_t kWorldSize = 2;
  int64_t n = 1LL << 19;

  FieldType field = std::get<0>(GetParam());
  auto dst_field = field;

  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());

  if (not signed_arith || msb == "Unknown") {
    return;
  }

  NdArrayRef prime = ring_zeros(field, {1});
  NdArrayRef input[2];
  input[0] = ring_rand(field, {n});
  input[1] = ring_zeros(field, {n});
  NdArrayRef msg = ring_zeros(field, {n});

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    prime.at<ring2k_t>(0) = GetPrime<ring2k_t>();
    ring2k_t _prime = NdArrayView<ring2k_t>(prime)[0];

    using sT = std::make_signed<ring2k_t>::type;
    std::uniform_int_distribution<ring2k_t> uniform(
        -static_cast<sT>(_prime) / 5, static_cast<sT>(_prime) / 5);

    std::default_random_engine rdv;
    NdArrayView<sT> xmsg(msg);
    NdArrayView<ring2k_t> rnd0(input[0]);
    NdArrayView<ring2k_t> rnd1(input[1]);

    for (int64_t i = 0; i < xmsg.numel(); ++i) {
      rnd0[i] %= _prime;

      xmsg[i] = uniform(rdv);
      rnd1[i] = xmsg[i] > 0 ? xmsg[i] : _prime - std::abs(xmsg[i]);

      rnd1[i] = (rnd1[i] + _prime - rnd0[i]) % _prime;
    }
  });

  bool do_trunc = msb == "One";
  const size_t _shft = 11;
  std::optional<size_t> shft = std::nullopt;
  if (do_trunc) {
    shft = _shft;
  }

  NdArrayRef oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);

    [[maybe_unused]] size_t sent0 = ctx->GetStats()->sent_bytes;
    oup[rank] = RingExtPrime(input[rank], prime, dst_field, shft, base);
    sent0 = ctx->GetStats()->sent_bytes - sent0;

    DISPATCH_ALL_FIELDS(field, "", [&]() {
      ring2k_t _prime = NdArrayView<ring2k_t>(prime)[0];
      SPDLOG_INFO(
          "Conv {}-bit prime to k={} ring, trunc = {}. Sent {} bits per "
          "conversion",
          std::ceil(std::log2(1. * _prime)), SizeOf(dst_field) * 8, do_trunc,
          sent0 * 8. / n);
    });
  });

  NdArrayRef _got = ring_add(oup[0], oup[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using sT0 = std::make_signed<ring2k_t>::type;
    NdArrayView<sT0> expected(msg);

    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using sT1 = std::make_signed<ring2k_t>::type;
      NdArrayView<sT1> got(_got);
      for (int64_t i = 0; i < n; ++i) {
        auto e = expected[i] >> (do_trunc ? *shft : 0);
        ASSERT_NEAR(e, got[i], 1);
      }
    });
  });
}

}  // namespace spu::mpc::cheetah
