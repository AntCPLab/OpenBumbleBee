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

#include <optional>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"
#include "seal/util/uintarithsmallmod.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetor::test {

void MaskItInplace(NdArrayRef a, size_t width) {
  auto field = a.eltype().as<RingTy>()->field();
  if (width == SizeOf(field) * 8) {
    return;
  }
  DISPATCH_ALL_FIELDS(field, "mask", [&]() {
    NdArrayView<ring2k_t> _a(a);
    ring2k_t msk = (static_cast<ring2k_t>(1) << width) - 1;
    pforeach(0, _a.numel(), [&](int64_t i) { _a[i] &= msk; });
  });
}

uint64_t GetPlainPrime(FieldType ft) {
  switch (ft) {
    case FM32:
      return 2147377153ULL;
    case FM64:
      return 1152921504606584833ULL;
    default:
    case FM128:
      return 1152921504606683137ULL;
  }
}

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

int64_t ToSignType(uint64_t x, const seal::Modulus &prime) {
  SPU_ENFORCE(x < prime.value());
  uint64_t half = prime.value() >> 1;
  if (x < half) {
    return x;
  }
  return static_cast<int64_t>(x - prime.value());
}

class UtilTest : public ::testing::TestWithParam<FieldType> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetor, UtilTest, testing::Values(FieldType::FM32, FieldType::FM64),
    [](const testing::TestParamInfo<UtilTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(UtilTest, ProbConvRing2kShareToPrimeShare_Basic) {
  const int64_t n = 1LL << 10;
  const auto field = GetParam();
  const auto prime = seal::Modulus(GetPlainPrime(field));

  int msg_width = 20;
  auto msg = ring_rand(field, {n});
  DISPATCH_ALL_FIELDS(field, "setup", [&]() {
    NdArrayView<ring2k_t> xmsg(msg);
    ring2k_t msk = (static_cast<ring2k_t>(1) << (msg_width - 1)) - 1;
    pforeach(0, n, [&](int64_t i) {
      xmsg[i] &= msk;
      if (i & 1) {
        // negative val
        xmsg[i] = -xmsg[i];
      }
    });
  });

  auto ashr0 = ring_rand(field, {n}).as(makeType<AShrTy>(field));
  auto ashr1 = ring_sub(msg, ashr0).as(makeType<AShrTy>(field));

  std::vector<uint64_t> hshr0(n);
  std::vector<uint64_t> hshr1(n);

  ProbConvRing2kShareToPrimeShare(ashr0, absl::MakeSpan(hshr0), prime, 0);
  ProbConvRing2kShareToPrimeShare(ashr1, absl::MakeSpan(hshr1), prime, 1);

  int64_t count_error = 0;
  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<const ring2k_t> expected(msg);
    using sT = std::make_signed<ring2k_t>::type;
    for (int64_t i = 0; i < expected.numel(); ++i) {
      sT got = ToSignType(seal::util::add_uint_mod(hshr0[i], hshr1[i], prime),
                          prime);
      sT exp = ToSignType(expected[i], sizeof(ring2k_t) * 8);
      count_error += (got != exp);
    }
  });
  ASSERT_NEAR(std::pow(2., (int)(msg_width - SizeOf(field) * 8) - 1),
              count_error * 1. / n, 1e-3);
}

TEST_P(UtilTest, ProbConvRing2kShareToPrimeShare_SpecificWidth) {
  const int64_t n = 1L << 10;
  const auto field = GetParam();
  size_t field_width = SizeOf(field) * 8 - 8;
  const auto prime = seal::Modulus(GetPlainPrime(field));

  int msg_width = 15;
  auto msg = ring_rand(field, {n});
  DISPATCH_ALL_FIELDS(field, "setup", [&]() {
    NdArrayView<ring2k_t> xmsg(msg);
    ring2k_t ring_msk = (static_cast<ring2k_t>(1) << field_width) - 1;
    ring2k_t msg_msk = (static_cast<ring2k_t>(1) << (msg_width - 1)) - 1;

    pforeach(0, n, [&](int64_t i) {
      xmsg[i] &= msg_msk;

      if (i & 1) {
        // negative val
        xmsg[i] = -xmsg[i];
      }

      xmsg[i] &= ring_msk;
    });
  });

  auto ashr0 = ring_rand(field, {n});
  MaskItInplace(ashr0, field_width);

  auto ashr1 = ring_sub(msg, ashr0);
  MaskItInplace(ashr1, field_width);

  ashr0 = ashr0.as(makeType<AShrTy>(field, field_width));
  ashr1 = ashr1.as(makeType<AShrTy>(field, field_width));

  std::vector<uint64_t> hshr0(n);
  std::vector<uint64_t> hshr1(n);

  ProbConvRing2kShareToPrimeShare(ashr0, absl::MakeSpan(hshr0), prime, 0);
  ProbConvRing2kShareToPrimeShare(ashr1, absl::MakeSpan(hshr1), prime, 1);

  int64_t count_error = 0;
  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<const ring2k_t> expected(msg);
    using sT = std::make_signed<ring2k_t>::type;
    for (int64_t i = 0; i < expected.numel(); ++i) {
      sT got = ToSignType(seal::util::add_uint_mod(hshr0[i], hshr1[i], prime),
                          prime);
      sT exp = ToSignType(expected[i], field_width);
      count_error += (got != exp);
    }
  });

  ASSERT_NEAR(std::pow(2., (int)(msg_width - field_width) - 1),
              count_error * 1. / n, 1e-3);
}

TEST_P(UtilTest, ProbConvRing2kShareToPrimeShare_LiftUp) {
  const int64_t n = 1L << 10;
  const auto field = GetParam();

  // To prevent prob error in ProbConvRing2kShareToPrimeShare (i.e,. convert
  // from large message). We first lift up the share to a larger ring.
  int msg_width = SizeOf(field) * 8 - 4;  // large message
  int ext_width = 16;  // larger margin width, smaller error prob
  auto msg = ring_rand(field, {n});
  DISPATCH_ALL_FIELDS(field, "setup", [&]() {
    NdArrayView<ring2k_t> xmsg(msg);
    ring2k_t msg_msk = (static_cast<ring2k_t>(1) << (msg_width - 1)) - 1;

    pforeach(0, n, [&](int64_t i) {
      xmsg[i] &= msg_msk;

      if (i & 1) {
        // negative val
        xmsg[i] = -xmsg[i];
      }
    });
  });

  NdArrayRef ashr[2];
  ashr[0] = ring_rand(field, {n});
  ashr[1] = ring_sub(msg, ashr[0]);

  NdArrayRef lifted_ashr[2];
  utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
    auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
        conn, CheetahOtKind::YACL_Softspoken);

    cheetah::RingExtendProtocol prot(base);
    cheetah::RingExtendProtocol::Meta meta;

    meta.src_ring = field;
    meta.src_width = SizeOf(field) * 8;
    meta.dst_ring = field == FM32 ? FM64 : FM128;
    meta.dst_width = msg_width + ext_width;
    meta.use_heuristic = true;
    meta.signed_arith = true;

    [[maybe_unused]] size_t b0 = ctx->GetStats()->sent_bytes;
    lifted_ashr[rank] = prot.Compute(ashr[rank], meta);
    [[maybe_unused]] size_t b1 = ctx->GetStats()->sent_bytes;
    SPDLOG_INFO("(Heuristic) Extend from {} bits to {} bits sent {} bits per",
                meta.src_width, meta.dst_width, (b1 - b0) * 8. / n);

    lifted_ashr[rank] =
        lifted_ashr[rank].as(makeType<AShrTy>(meta.dst_ring, meta.dst_width));
  });

  DISPATCH_ALL_FIELDS(field, "check_lift", [&]() {
    auto dst_ring = field == FM32 ? FM64 : FM128;
    NdArrayView<const ring2k_t> expected(msg);
    using sT0 = std::make_signed<ring2k_t>::type;

    DISPATCH_ALL_FIELDS(dst_ring, "check_lift", [&]() {
      using sT1 = std::make_signed<ring2k_t>::type;
      auto _got = ring_add(lifted_ashr[0], lifted_ashr[1]);
      MaskItInplace(_got, msg_width + ext_width);
      NdArrayView<const ring2k_t> got(_got);

      for (int64_t i = 0; i < expected.numel(); ++i) {
        sT0 exp = ToSignType(expected[i], SizeOf(field) * 8);
        sT1 g = ToSignType(got[i], msg_width + ext_width);
        ASSERT_EQ(exp, g);
      }
    });
  });

  const auto prime = seal::Modulus(GetPlainPrime(field));

  std::vector<uint64_t> hshr0(n);
  std::vector<uint64_t> hshr1(n);
  ProbConvRing2kShareToPrimeShare(lifted_ashr[0], absl::MakeSpan(hshr0), prime,
                                  0);
  ProbConvRing2kShareToPrimeShare(lifted_ashr[1], absl::MakeSpan(hshr1), prime,
                                  1);

  int64_t count_error = 0;
  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<const ring2k_t> expected(msg);
    using sT = std::make_signed<ring2k_t>::type;
    for (int64_t i = 0; i < expected.numel(); ++i) {
      sT got = ToSignType(seal::util::add_uint_mod(hshr0[i], hshr1[i], prime),
                          prime);
      sT exp = ToSignType(expected[i], SizeOf(field) * 8);
      count_error += (got != exp);
    }
  });

  ASSERT_NEAR(std::pow(2., -ext_width), count_error * 1. / n, 1e-3);
}

TEST_P(UtilTest, ProbConvRing2kShareToPrimeShare_CastDown) {
  const auto field = GetParam();
  if (field == FM32) {
    return;
  }

  // Cast from large ring to smaller field
  // Pr(error) <= 2^{24 - k} for k = 64/128
  const int msg_width = 24;
  const int64_t n = 1L << 18;
  auto msg = ring_rand(field, {n});
  DISPATCH_ALL_FIELDS(field, "setup", [&]() {
    NdArrayView<ring2k_t> xmsg(msg);
    ring2k_t msg_msk = (static_cast<ring2k_t>(1) << (msg_width - 1)) - 1;

    pforeach(0, n, [&](int64_t i) {
      xmsg[i] &= msg_msk;

      if (i & 1) {
        // negative val
        xmsg[i] = -xmsg[i];
      }
    });
  });

  auto target_ft = FM32;

  auto ashr0 = ring_rand(field, {n}).as(makeType<AShrTy>(field));
  auto ashr1 = ring_sub(msg, ashr0).as(makeType<AShrTy>(field));

  std::vector<uint64_t> hshr0(n);
  std::vector<uint64_t> hshr1(n);

  seal::Modulus prime = GetPlainPrime(target_ft);
  ProbConvRing2kShareToPrimeShare(ashr0, absl::MakeSpan(hshr0), prime, 0);
  ProbConvRing2kShareToPrimeShare(ashr1, absl::MakeSpan(hshr1), prime, 1);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<const ring2k_t> expected(msg);
    using sT = std::make_signed<ring2k_t>::type;
    for (int64_t i = 0; i < expected.numel(); ++i) {
      sT got = ToSignType(seal::util::add_uint_mod(hshr0[i], hshr1[i], prime),
                          prime);
      sT exp = ToSignType(expected[i], SizeOf(field) * 8);
      ASSERT_EQ(exp, got);
    }
  });
}

}  // namespace spu::mpc::cheetor::test
