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

#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"

#include <memory>
#include <optional>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"
#include "seal/util/rns.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetor::test {

class CheetorMulProtTest : public ::testing::TestWithParam<FieldType> {
 public:
  static constexpr int64_t kBenchSize = 1LL << 10;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetor, CheetorMulProtTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<CheetorMulProtTest::ParamType>& p) {
      return fmt::format("{}", p.param);
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

TEST_P(CheetorMulProtTest, Basic) {
  int64_t numel = kBenchSize;
  auto field = GetParam();

  auto msg_x = ring_rand(field, {numel / 4, 4});
  auto msg_y = ring_rand(field, {numel / 4, 4});
  ring_lshift_(msg_x, SizeOf(field) * 8 - 10);
  ring_rshift_(msg_x, SizeOf(field) * 8 - 10);
  ring_lshift_(msg_y, SizeOf(field) * 8 - 10);
  ring_rshift_(msg_y, SizeOf(field) * 8 - 10);

  NdArrayRef xshr[2];
  NdArrayRef yshr[2];
  xshr[0] = ring_rand(field, msg_x.shape()).as(makeType<AShrTy>(field));
  xshr[1] = ring_sub(msg_x, xshr[0]).as(makeType<AShrTy>(field));
  yshr[0] = ring_rand(field, msg_y.shape()).as(makeType<AShrTy>(field));
  yshr[1] = ring_sub(msg_y, yshr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    CheetorMulProt prot(conn);
    prot.LazyInit(field);

    [[maybe_unused]] size_t he_sent = conn->GetStats()->sent_bytes;
    outp[rank] = prot.RingShareToPrimeShareMul(xshr[rank], yshr[rank]);
    he_sent = conn->GetStats()->sent_bytes - he_sent;

    auto comm = std::make_shared<Communicator>(conn);
    auto ot_base = std::make_shared<cheetah::BasicOTProtocols>(
        comm, CheetahOtKind::YACL_Softspoken);
    PrimeRingCastProtocol::Meta meta;
    meta.prime = CheetorMulProt::GetPrimesProduct(field);
    meta.prime_width = CheetorMulProt::PrimeNumBits(field);
    meta.dst_width = SizeOf(field) * 8;
    meta.dst_ring = field;
    meta.truncate_nbits = std::nullopt;

    [[maybe_unused]] size_t ext_sent = conn->GetStats()->sent_bytes;
    PrimeRingCastProtocol prime_prot(ot_base);
    outp[rank] = prime_prot.Compute(outp[rank], meta);
    ext_sent = conn->GetStats()->sent_bytes - ext_sent;

    SPDLOG_INFO("Mul {} {} + {} bits per", xshr[0].eltype(),
                he_sent * 8. / numel, ext_sent * 8. / numel);
  });
  SPU_ENFORCE_EQ(msg_y.shape(), outp[0].shape());

  auto got = ring_add(outp[0], outp[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<ring2k_t> x_msg(msg_x);
    NdArrayView<ring2k_t> y_msg(msg_y);
    NdArrayView<ring2k_t> _got(got);

    for (int64_t i = 0; i < numel; ++i) {
      ASSERT_NEAR(x_msg[i] * y_msg[i], _got[i], 1);
    }
  });
}

TEST_P(CheetorMulProtTest, Square) {
  int64_t numel = kBenchSize;
  auto field = GetParam();

  NdArrayRef ring2k_shr[2];

  auto rnd_msg = ring_rand(field, {numel / 8, 8});
  ring_lshift_(rnd_msg, SizeOf(field) * 8 - 10);
  ring_rshift_(rnd_msg, SizeOf(field) * 8 - 10);

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    CheetorMulProt prot(conn);
    prot.LazyInit(field);

    [[maybe_unused]] size_t he_sent = conn->GetStats()->sent_bytes;
    outp[rank] = prot.RingShareToPrimeShareSquare(ring2k_shr[rank]);
    he_sent = conn->GetStats()->sent_bytes - he_sent;

    auto comm = std::make_shared<Communicator>(conn);
    auto ot_base = std::make_shared<cheetah::BasicOTProtocols>(
        comm, CheetahOtKind::YACL_Softspoken);
    PrimeRingCastProtocol::Meta meta;
    meta.prime = CheetorMulProt::GetPrimesProduct(field);
    meta.prime_width = CheetorMulProt::PrimeNumBits(field);
    meta.dst_width = SizeOf(field) * 8;
    meta.dst_ring = field;
    meta.truncate_nbits = std::nullopt;

    [[maybe_unused]] size_t ext_sent = conn->GetStats()->sent_bytes;
    PrimeRingCastProtocol prime_prot(ot_base);
    outp[rank] = prime_prot.Compute(outp[rank], meta);
    ext_sent = conn->GetStats()->sent_bytes - ext_sent;

    SPDLOG_DEBUG("Square {} {} + {} bits per", ring2k_shr[0].eltype(),
                 he_sent * 8. / numel, ext_sent * 8. / numel);
  });
  SPU_ENFORCE_EQ(rnd_msg.shape(), outp[0].shape());

  auto got = ring_add(outp[0], outp[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<ring2k_t> _msg(rnd_msg);
    NdArrayView<ring2k_t> _got(got);

    for (int64_t i = 0; i < numel; ++i) {
      ASSERT_EQ(_msg[i] * _msg[i], _got[i]);
    }
  });
}

NdArrayRef CastDownRing(const NdArrayRef& in, const FieldType& ftype) {
  SPU_ENFORCE(in.eltype().isa<AShrTy>());
  const auto field = in.eltype().as<AShrTy>()->field();
  const auto numel = in.numel();
  const size_t k = SizeOf(field) * 8;
  const size_t to_bits = SizeOf(ftype) * 8;
  if (to_bits == k) {
    // euqal ring size, do nothing
    return in;
  } else if (to_bits < k) {
    // cast down is a local procedure
    return DISPATCH_ALL_FIELDS(field, "cheetah.castdown", [&]() {
      using from_ring2k_t = ring2k_t;
      return DISPATCH_ALL_FIELDS(ftype, "cheetah.castdown", [&]() {
        using to_ring2k_t = ring2k_t;
        NdArrayRef res(makeType<AShrTy>(ftype), in.shape());
        NdArrayView<const from_ring2k_t> _in(in);
        NdArrayView<to_ring2k_t> _res(res);
        pforeach(0, numel, [&](int64_t idx) {
          _res[idx] = static_cast<to_ring2k_t>(_in[idx]);
        });
        return res;
      });
    });
  }

  SPU_THROW("Cast down only");
}

TEST_P(CheetorMulProtTest, Pow4) {
  int64_t numel = kBenchSize;
  auto field = GetParam();

  NdArrayRef ring2k_shr[2];

  auto rnd_msg = ring_rand(field, {numel / 8, 8});
  int fxp = 10;
  ring_lshift_(rnd_msg, SizeOf(field) * 8 - 13);
  ring_rshift_(rnd_msg, SizeOf(field) * 8 - 13);

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  NdArrayRef square[2];
  NdArrayRef cubic[2];
  NdArrayRef quad[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    FieldType work_ft = FM32;
    PrimeRingCastProtocol::Meta meta;
    meta.prime = CheetorMulProt::GetPrimesProduct(work_ft);
    meta.prime_width = CheetorMulProt::PrimeNumBits(work_ft);
    meta.dst_ring = work_ft;
    meta.dst_width = SizeOf(work_ft) * 8;
    meta.truncate_nbits = fxp;

    auto comm = std::make_shared<Communicator>(conn);
    auto ot_base = std::make_shared<cheetah::BasicOTProtocols>(
        comm, CheetahOtKind::YACL_Ferret);

    PrimeRingCastProtocol prime_cast_prot(ot_base);
    CheetorMulProt mul_prot(conn);

    mul_prot.LazyInit(meta.dst_ring);

    size_t sent = conn->GetStats()->sent_bytes;
    // x mod 2^k => x mod p
    auto _x_hshr = std::vector<uint64_t>(numel);
    auto _x2_hshr = std::vector<uint64_t>(numel);
    auto x_hshr = absl::MakeSpan(_x_hshr);
    auto x2_hshr = absl::MakeSpan(_x2_hshr);

    int64_t stride = sizeof(uint64_t) / SizeOf(work_ft);
    auto oshape = ring2k_shr[rank].shape();

    ProbConvRing2kShareToPrimeShare(
        ring2k_shr[rank], x_hshr, CheetorMulProt::GetWorkingPrimes(work_ft)[0],
        rank);
    auto square_mod_p = mul_prot.PrimeShareSquare(x_hshr, work_ft);

    if (stride > 1) {
      square_mod_p = NdArrayRef(square_mod_p.buf(), square_mod_p.eltype(),
                                square_mod_p.shape(), {stride}, 0);
    }

    square[rank] = prime_cast_prot.Compute(square_mod_p, meta);

    // x^2 mod p => x^4 mod p
    ProbConvRing2kShareToPrimeShare(
        square[rank], x2_hshr, CheetorMulProt::GetWorkingPrimes(work_ft)[0],
        rank);

    auto quad_mod_p = mul_prot.PrimeShareSquare(x2_hshr, work_ft);

    if (stride > 1) {
      quad_mod_p = NdArrayRef(quad_mod_p.buf(), square_mod_p.eltype(),
                              square_mod_p.shape(), {stride}, 0);
    }

    quad[rank] = prime_cast_prot.Compute(quad_mod_p, meta);

    auto cubic_mod_p = mul_prot.PrimeShareMul(x_hshr, x2_hshr, work_ft)
                           .as(makeType<PrimeShrTy>(work_ft));
    if (stride > 1) {
      cubic_mod_p = NdArrayRef(cubic_mod_p.buf(), square_mod_p.eltype(),
                               square_mod_p.shape(), {stride}, 0);
    }

    meta.dst_ring = field;
    meta.dst_width = SizeOf(field) * 8;

    if (field != work_ft) {
      square[rank] = prime_cast_prot.Compute(square_mod_p, meta);
      quad[rank] = prime_cast_prot.Compute(quad_mod_p, meta);
    }
    cubic[rank] = prime_cast_prot.Compute(cubic_mod_p, meta);

    sent = conn->GetStats()->sent_bytes - sent;
    SPDLOG_INFO("Pow4 from {} sent {} MiB, {} bits per", field,
                sent / 1024. / 1024., sent * 8. / numel);
  });

  auto _got2 = ring_add(square[0], square[1]);
  auto _got3 = ring_add(cubic[0], cubic[1]);
  auto _got4 = ring_add(quad[0], quad[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _msg(rnd_msg);
    NdArrayView<sT> got2(_got2);
    NdArrayView<sT> got3(_got3);
    NdArrayView<sT> got4(_got4);

    double max_e2 = 0;
    double max_e3 = 0;
    double max_e4 = 0;

    for (int64_t i = 0; i < numel; ++i) {
      double e2 = std::pow(_msg[i] / std::pow(2., fxp), 2.);
      double e3 = std::pow(_msg[i] / std::pow(2., fxp), 3.);
      double e4 = std::pow(_msg[i] / std::pow(2., fxp), 4.);

      double g2 = got2[i] / std::pow(2., fxp);
      double g3 = got3[i] / std::pow(2., fxp);
      double g4 = got4[i] / std::pow(2., fxp);

      max_e2 = std::max(max_e2, std::abs(e2 - g2));
      max_e3 = std::max(max_e3, std::abs(e3 - g3));
      max_e4 = std::max(max_e4, std::abs(e4 - g4));
      // if (i < 8) {
      //   printf("%f => %f\t%f => %f\t%f => %f\n", e2, g2, e3, g3, e4, g4);
      // }
    }
    printf("%f %f %f\n", max_e2, max_e3, max_e4);
  });
}

TEST_P(CheetorMulProtTest, ScaledSineFromRing) {
  int64_t numel = kBenchSize;
  auto field = GetParam();

  NdArrayRef ring2k_shr[2];

  auto rnd_msg = ring_rand(field, {numel / 8, 8});
  // L = 4, f = 8
  ring_lshift_(rnd_msg, SizeOf(field) * 8 - 12);
  ring_rshift_(rnd_msg, SizeOf(field) * 8 - 12);

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  int fxp_inp = 13;
  int fxp_mul = 14;  // prim32 at most 31 - 2bits (-2 is due to prime2ring)
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    CheetorMulProt prot(conn);
    prot.LazyInit(field);

    const uint32_t prime32 = CheetorMulProt::GetWorkingPrimes(FM32)[0];

    NdArrayRef sin_part = ring_zeros(FM64, ring2k_shr[rank].shape());
    NdArrayRef cos_part = ring_zeros(FM64, ring2k_shr[rank].shape());

    DISPATCH_ALL_FIELDS(field, "local.conv", [&]() {
      ring2k_t msk = (static_cast<ring2k_t>(1) << fxp_inp) - 1;

      NdArrayView<const ring2k_t> ashr(ring2k_shr[rank]);
      NdArrayView<uint64_t> sin(sin_part);
      NdArrayView<uint64_t> cos(cos_part);

      for (int64_t i = 0; i < numel; ++i) {
        double fraction = static_cast<double>(ashr[i] & msk) / (msk + 1);
        double ss = std::sin(2. * M_PI * fraction) * (1L << fxp_mul);
        double cc = std::cos(2. * M_PI * fraction) * (1L << fxp_mul);

        sin[i] = std::abs(ss);
        if (std::signbit(ss) and sin[i] > 0) {
          sin[i] = prime32 - sin[i];
        }

        cos[i] = std::abs(cc);
        if (std::signbit(cc) and cos[i] > 0) {
          cos[i] = prime32 - cos[i];
        }
      }
    });

    auto sin_span = absl::MakeConstSpan(&sin_part.at<uint64_t>(0), numel);
    auto cos_span = absl::MakeConstSpan(&cos_part.at<uint64_t>(0), numel);

    size_t he_sent = conn->GetStats()->sent_bytes;
    auto mix0 = prot.MulToPrime(rank == 0 ? sin_span : cos_span, FM32);
    auto mix1 = prot.MulToPrime(rank == 1 ? sin_span : cos_span, FM32);
    he_sent = conn->GetStats()->sent_bytes - he_sent;

    NdArrayView<uint32_t> added(mix0);
    NdArrayView<const uint32_t> op0(mix1);
    // mix0 + mix1
    pforeach(0, numel, [&](int64_t i) {
      added[i] += op0[i];
      added[i] -= (added[i] < prime32 ? 0 : prime32);
    });

    auto comm = std::make_shared<Communicator>(conn);
    auto ot_base = std::make_shared<cheetah::BasicOTProtocols>(
        comm, CheetahOtKind::YACL_Ferret);

    size_t conv_sent = conn->GetStats()->sent_bytes;

    PrimeRingCastProtocol::Meta meta;
    meta.prime = CheetorMulProt::GetPrimesProduct(FM32);
    meta.prime_width = CheetorMulProt::PrimeNumBits(FM32);
    meta.dst_width = SizeOf(field) * 8;
    meta.dst_ring = field;
    meta.truncate_nbits = 2 * fxp_mul - fxp_inp;

    PrimeRingCastProtocol prime_prot(ot_base);
    outp[rank] = prime_prot.Compute(mix0, meta);

    conv_sent = conn->GetStats()->sent_bytes - conv_sent;
    SPDLOG_INFO(
        "Compute sin(2pi*x) on {} points from {} sent {} + {} bytes per", numel,
        field, he_sent * 1. / numel, conv_sent * 1. / numel);
  });
  // SPU_ENFORCE_EQ(rnd_msg.shape(), outp[0].shape());

  auto got = ring_add(outp[0], outp[1]);
  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _msg(rnd_msg);
    NdArrayView<sT> _got(got);
    double scale = std::pow(2., fxp_inp);

    double max_err = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double expected = std::sin(2. * M_PI * _msg[i] / scale);
      double got = _got[i] / scale;
      max_err = std::max(max_err, std::abs(expected - got));
    }
    printf("max error %f\n", max_err);
  });
}

}  // namespace spu::mpc::cheetor::test
