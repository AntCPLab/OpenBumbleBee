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

#include "libspu/mpc/cheetah/alg.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/dialect/cheetor/state.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahAlgoTest : public ::testing::TestWithParam<FieldType> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetor, CheetahAlgoTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<CheetahAlgoTest::ParamType>& p) {
      return fmt::format("{}", p.param);
    });

TEST_P(CheetahAlgoTest, BatchLessThan) {
  FieldType field = GetParam();
  int fxp = 18;
  Shape shape = {10, 20, 30};
  int64_t n = shape.numel();

  NdArrayRef rnd[2];
  rnd[0] = ring_rand(field, shape);
  std::vector<double> msg(n);

  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-8., 8);
  std::generate_n(msg.data(), msg.size(), [&]() { return uniform(rdv); });
  DISPATCH_ALL_FIELDS(field, "msg", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    rnd[1] = ring_zeros(field, shape);
    NdArrayView<sT> r0(rnd[0]);
    NdArrayView<sT> r1(rnd[1]);
    pforeach(0, n, [&](int64_t i) {
      r1[i] = static_cast<sT>(msg[i] * (1L << fxp)) - r0[i];
    });

    rnd[0].as(makeType<AShrTy>(field));
    rnd[1].as(makeType<AShrTy>(field));
  });

  std::vector<NdArrayRef> comp[2];
  std::vector<float> pl = {-3.0, -1.95, 3.};
  spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig conf;
        conf.set_field(field);
        conf.set_fxp_fraction_bits(fxp);
        std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, lctx);
        KernelEvalContext kcontext(obj.get());

        int rnk = lctx->Rank();
        size_t sent = lctx->GetStats()->sent_bytes;
        size_t recv = lctx->GetStats()->recv_bytes;

        comp[rnk] = BatchLessThan(&kcontext, rnd[rnk], absl::MakeSpan(pl));

        sent = lctx->GetStats()->sent_bytes - sent;
        recv = lctx->GetStats()->recv_bytes - recv;
        printf("BatchLessThan %lld elements sent %f KiB recv %f KiB\n", n,
               sent / 1024., recv / 1024.);
      });

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    // bool allow_tol = TestEnvFlag(EnvFlag::SPU_BB_ENABLE_APPROX_LESS_THAN);
    [[maybe_unused]] double tol = 1 / 16.;  // keep 4bits
    for (size_t b = 0; b < pl.size(); ++b) {
      auto _got = ring_xor(comp[0][b], comp[1][b]);
      NdArrayView<ring2k_t> got(_got);
      for (int64_t i = 0; i < n; ++i) {
        auto _g = got[i] == 1;

        if (_g) {
          EXPECT_LT(msg[i], pl[b]);
        } else {
          EXPECT_GT(msg[i], pl[b]);
        }
      }
    }
  });
}

TEST_P(CheetahAlgoTest, Square) {
  int64_t numel = 128;
  FieldType field = GetParam();

  NdArrayRef ring2k_shr[2];

  auto rnd_msg = ring_rand(field, {numel / 8, 8});

  ring_lshift_(rnd_msg, SizeOf(field) * 8 - (field == FM32 ? 8 : 16));
  ring_rshift_(rnd_msg, SizeOf(field) * 8 - (field == FM32 ? 8 : 16));

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    RuntimeConfig conf;
    conf.set_field(field);
    conf.set_fxp_fraction_bits(16);
    // conf.mutable_cheetah_2pc_config()->set_enable_cheetor(true);
    std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, conn);
    KernelEvalContext kcontext(obj.get());

    outp[rank] =
        cheetor::MulThenTrunc(&kcontext, ring2k_shr[rank], ring2k_shr[rank],
                              field, /*trunc*/ 0, /*keep_ft*/ true);
  });

  SPU_ENFORCE_EQ(rnd_msg.shape(), outp[0].shape());

  auto got = ring_add(outp[0], outp[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    NdArrayView<ring2k_t> _msg(rnd_msg);
    NdArrayView<ring2k_t> _got(got);

    for (int64_t i = 0; i < numel; ++i) {
      ASSERT_NEAR(_msg[i] * _msg[i], _got[i], 1);
    }
  });
}

TEST_P(CheetahAlgoTest, NExp) {
  int64_t numel = 128;
  FieldType field = GetParam();

  NdArrayRef ring2k_shr[2];

  std::uniform_real_distribution<double> dist(-8.0, 0.0);
  std::default_random_engine rd;
  std::vector<double> real_vec(numel);
  for (int64_t i = 0; i < numel; ++i) {
    real_vec[i] = dist(rd);
  }

  const int fxp = 16;

  auto rnd_msg = ring_zeros(field, {numel});
  DISPATCH_ALL_FIELDS(field, "msg", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> xmsg(rnd_msg);
    pforeach(0, numel, [&](int64_t i) {
      xmsg[i] = std::round(real_vec[i] * (1L << fxp));
    });
  });

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef outp[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    RuntimeConfig conf;
    conf.set_field(field);
    conf.set_fxp_fraction_bits(fxp);
    // conf.mutable_cheetah_2pc_config()->set_enable_cheetor(true);
    conf.mutable_cheetah_2pc_config()->set_ot_kind(
        CheetahOtKind::YACL_Softspoken);
    std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, conn);
    KernelEvalContext kcontext(obj.get());
    auto cheetor_mul_prot =
        kcontext.getState<cheetor::CheetorMulState>()->GetMulProt();
    cheetor_mul_prot->LazyInit(FM64);

    size_t bytes = conn->GetStats()->sent_bytes;
    size_t action = conn->GetStats()->sent_actions;
    outp[rank] = cheetor::NExp_8(&kcontext, ring2k_shr[rank], fxp);
    bytes = conn->GetStats()->sent_bytes - bytes;
    action = conn->GetStats()->sent_actions - action;
    SPDLOG_INFO("Nexp ({}) for n = {}, sent {} MiB ({} B per), actions {}",
                field, numel, bytes * 1. / 1024. / 1024., bytes * 1. / numel,
                action);
  });

  SPU_ENFORCE_EQ(rnd_msg.shape(), outp[0].shape());

  auto got = ring_add(outp[0], outp[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _got(got);

    double max_err = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double expected = std::exp(real_vec[i]);
      double got = static_cast<double>(_got[i]) / (1L << fxp);
      max_err = std::max(max_err, std::abs(expected - got));
    }
    ASSERT_LE(max_err, 1e-3);
  });
}

}  // namespace spu::mpc::cheetah::test
