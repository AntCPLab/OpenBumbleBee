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

#include "libspu/mpc/cheetah/kernels/algo.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/env.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah::test {

class CheetahAlgoTest : public ::testing::Test {};

TEST_F(CheetahAlgoTest, Power4Int) {
  FieldType field = FM64;
  int64_t n = 1024;

  NdArrayRef rnd[2];
  rnd[0] = ring_rand(field, {n});
  rnd[1] = ring_rand(field, {n});
  rnd[0].as(makeType<AShrTy>(field));
  rnd[1].as(makeType<AShrTy>(field));

  NdArrayRef square[2];
  NdArrayRef quad[2];
  NdArrayRef cub[2];

  spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig conf;
        conf.set_field(field);
        std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, lctx);
        KernelEvalContext kcontext(obj.get());

        int rnk = lctx->Rank();
        auto [s, c, q] = ComputeUptoPower4(&kcontext, rnd[rnk], false);
        square[rnk] = s;
        quad[rnk] = q;
        cub[rnk] = c;
      });

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    auto m = ring_add(rnd[0], rnd[1]);
    auto s = ring_add(square[0], square[1]);
    auto q = ring_add(quad[0], quad[1]);
    auto c = ring_add(cub[0], cub[1]);

    NdArrayView<ring2k_t> _m(m);
    NdArrayView<ring2k_t> _s(s);
    NdArrayView<ring2k_t> _c(c);
    NdArrayView<ring2k_t> _q(q);

    for (int i = 0; i < n; ++i) {
      ring2k_t expect2 = _m[i] * _m[i];
      ring2k_t expect3 = expect2 * _m[i];
      ring2k_t expect4 = expect2 * expect2;

      EXPECT_NEAR(expect2, _s[i], 1);
      EXPECT_NEAR(expect3, _c[i], 1);
      EXPECT_NEAR(expect4, _q[i], 1);
    }
  });
}

TEST_F(CheetahAlgoTest, Power4Fxp) {
  FieldType field = FM64;
  int fxp = 18;
  int64_t n = 81920;

  NdArrayRef rnd[2];
  rnd[0] = ring_rand(field, {n});
  std::vector<double> msg(n);

  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-3., 3);
  std::generate_n(msg.data(), msg.size(), [&]() { return uniform(rdv); });
  DISPATCH_ALL_FIELDS(field, "msg", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    rnd[1] = ring_zeros(field, {n});
    NdArrayView<sT> r0(rnd[0]);
    NdArrayView<sT> r1(rnd[1]);
    pforeach(0, n, [&](int64_t i) {
      r1[i] = static_cast<sT>(msg[i] * (1L << fxp)) - r0[i];
    });

    rnd[0].as(makeType<AShrTy>(field));
    rnd[1].as(makeType<AShrTy>(field));
  });

  NdArrayRef square[2];
  NdArrayRef quad[2];
  NdArrayRef cub[2];

  spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig conf;
        conf.set_field(field);
        conf.set_fxp_fraction_bits(fxp);
        std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, lctx);
        KernelEvalContext kcontext(obj.get());

        int rnk = lctx->Rank();
        auto [s, c, q] = ComputeUptoPower4(&kcontext, rnd[rnk]);

        square[rnk] = s;
        quad[rnk] = q;
        cub[rnk] = c;
      });

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    auto s = ring_add(square[0], square[1]);
    auto q = ring_add(quad[0], quad[1]);
    auto c = ring_add(cub[0], cub[1]);
    using sT = std::make_signed<ring2k_t>::type;

    NdArrayView<sT> _s(s);
    NdArrayView<sT> _c(c);
    NdArrayView<sT> _q(q);
    const double scale = 1L << fxp;

    for (int i = 0; i < n; ++i) {
      double expect2 = std::pow(msg[i], 2);
      double expect3 = expect2 * msg[i];
      double expect4 = expect2 * expect2;

      double got2 = _s[i] / scale;
      double got3 = _c[i] / scale;
      double got4 = _q[i] / scale;

      ASSERT_NEAR(expect2, got2, 0.005);
      ASSERT_NEAR(expect3, got3, 0.005);
      ASSERT_NEAR(expect4, got4, 0.005);
    }
  });
}

TEST_F(CheetahAlgoTest, BatchLessThan) {
  FieldType field = FM64;
  int fxp = 18;
  int64_t n = 1 << 18;

  NdArrayRef rnd[2];
  rnd[0] = ring_rand(field, {n});
  std::vector<double> msg(n);

  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-8., 8);
  std::generate_n(msg.data(), msg.size(), [&]() { return uniform(rdv); });
  DISPATCH_ALL_FIELDS(field, "msg", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    rnd[1] = ring_zeros(field, {n});
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
    bool allow_tol = TestEnvFlag(EnvFlag::SPU_CHEETAH_ENABLE_APPROX_LESS_THAN);
    double tol = 1 / 16.;  // keep 4bits
    for (size_t b = 0; b < pl.size(); ++b) {
      auto _got = ring_xor(comp[0][b], comp[1][b]);
      NdArrayView<ring2k_t> got(_got);
      for (int64_t i = 0; i < n; ++i) {
        auto _g = got[i] == 1;
        if (allow_tol && std::abs(msg[i] - pl[b]) < tol) {
          continue;
        }

        if (_g) {
          EXPECT_LT(msg[i], pl[b]);
        } else {
          EXPECT_GT(msg[i], pl[b]);
        }
      }
    }
  });
}

TEST_F(CheetahAlgoTest, RingChange) {
  FieldType field = FM64;
  FieldType down_field = FM32;
  FieldType up_field = FM128;

  const int fxp = 18;
  int64_t n = 8192;

  NdArrayRef rnd[2];
  rnd[0] = ring_rand(field, {n});
  std::vector<double> msg(n);

  std::default_random_engine rdv;
  std::uniform_real_distribution<double> uniform(-1., 1.);
  std::generate_n(msg.data(), msg.size(), [&]() { return uniform(rdv); });

  DISPATCH_ALL_FIELDS(field, "msg", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    rnd[1] = ring_zeros(field, {n});
    NdArrayView<sT> r0(rnd[0]);
    NdArrayView<sT> r1(rnd[1]);
    pforeach(0, n, [&](int64_t i) {
      r1[i] = static_cast<sT>(msg[i] * (1L << fxp)) - r0[i];
    });

    rnd[0].as(makeType<AShrTy>(field));
    rnd[1].as(makeType<AShrTy>(field));
  });

  NdArrayRef down[2];
  NdArrayRef up[2];
  spu::mpc::utils::simulate(
      2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
        RuntimeConfig conf;
        conf.set_field(field);
        conf.set_fxp_fraction_bits(fxp);
        std::shared_ptr<SPUContext> obj = makeCheetahProtocol(conf, lctx);
        KernelEvalContext kcontext(obj.get());

        int rnk = lctx->Rank();
        down[rnk] = ChangeRing(&kcontext, rnd[rnk], down_field);
        up[rnk] = ChangeRing(&kcontext, rnd[rnk], up_field);
      });

  DISPATCH_ALL_FIELDS(up_field, "check_up", [&]() {
    auto u = ring_add(up[0], up[1]);
    using sT = std::make_signed<ring2k_t>::type;

    NdArrayView<sT> _u(u);
    const double scale = 1L << fxp;

    for (int i = 0; i < n; ++i) {
      double expect = msg[i];
      double got = _u[i] / scale;

      ASSERT_NEAR(expect, got, 1. / (1L << fxp));
    }
  });

  DISPATCH_ALL_FIELDS(down_field, "check_down", [&]() {
    auto u = ring_add(down[0], down[1]);
    using sT = std::make_signed<ring2k_t>::type;

    NdArrayView<sT> _u(u);
    const double scale = 1L << fxp;

    for (int i = 0; i < n; ++i) {
      double expect = msg[i];
      double got = _u[i] / scale;

      ASSERT_NEAR(expect, got, 1. / (1L << fxp));
    }
  });
}

}  // namespace spu::mpc::cheetah::test
