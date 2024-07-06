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

#include "libspu/kernel/hal/intrinsic/nn/activation.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/core/xt_helper.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::kernel::hal::test {

class ActivationTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ActivationTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(false)),
    [](const testing::TestParamInfo<ActivationTest::ParamType>& p) {
      return fmt::format("{}xCheetor_{}", std::get<0>(p.param),
                         (std::get<1>(p.param) ? "enabled" : "disabled"));
    });

template <typename T>
spu::Value infeed(spu::SPUContext* hctx, const xt::xarray<T>& ds) {
  spu::device::ColocatedIo cio(hctx);
  if (hctx->lctx()->Rank() == 0) {
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), ds);
  }
  cio.sync();
  auto x = cio.deviceGetVar("x-0");
  return x;
}

TEST_P(ActivationTest, Seg3Gelu) {
  using namespace spu::mpc;
  FieldType field = std::get<0>(GetParam());
  [[maybe_unused]] bool enable_cheetor = std::get<1>(GetParam());
  Shape shape = {10, 50, 20};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-7.0, 7.0);
  std::uniform_real_distribution<double> uniform_large(-16.0, 16.0);

  std::vector<size_t> _shape;
  for (int64_t d : shape) {
    _shape.push_back((size_t)d);
  }
  xt::xarray<double> _x(_shape);
  size_t numel = _x.size();
  std::generate_n(_x.data(), numel / 2, [&]() { return uniform(rdv); });
  std::generate_n(_x.data() + numel / 2, numel - numel / 2,
                  [&]() { return uniform_large(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(field == FM32 ? 12 : 18);
    rt_config.set_enable_hal_profile(true);
    rt_config.set_enable_pphlo_profile(true);
    rt_config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
    // keep 4bits for approx less
    rt_config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto x = infeed<double>(ctx, _x);

    size_t bytes_sent = lctx->GetStats()->sent_bytes;
    auto gelu = hal::intrinsic::nn::f_seg3_gelu(ctx, x);
    bytes_sent = lctx->GetStats()->sent_bytes - bytes_sent;

    gelu = hlo::Reveal(ctx, gelu);
    if (lctx->Rank() == 0) {
      return;
    }

    double fxp = std::pow(2., rt_config.fxp_fraction_bits());
    double max_err = 0.;

    DISPATCH_ALL_FIELDS(field, "check", [&]() {
      using sT = std::make_signed<ring2k_t>::type;
      for (int64_t i = 0; i < gelu.numel(); ++i) {
        double expected =
            0.5 * _x[i] *
            (1 + std::tanh(std::sqrt(2. / M_PI) *
                           (_x[i] + 0.044715 * std::pow(_x[i], 3))));
        // double expected = std::pow(_x[i], 4);
        double got = gelu.data().at<sT>(i) / fxp;

        double e = std::abs(got - expected);
        if (e > max_err) {
          max_err = e;
        }
      }
    });

    printf("seg3_gelu max error %f\n", max_err);
  });
}

TEST_P(ActivationTest, Seg4Silu) {
  using namespace spu::mpc;
  FieldType field = std::get<0>(GetParam());
  if (field == FM32) {
    // FM32 seems not working for seg4silu
    return;
  }

  [[maybe_unused]] bool enable_cheetor = std::get<1>(GetParam());
  Shape shape = {100, 50, 10};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-7.0, 7.0);
  std::uniform_real_distribution<double> uniform_large(-16.0, 16.0);

  std::vector<size_t> _shape;
  for (int64_t d : shape) {
    _shape.push_back((size_t)d);
  }

  xt::xarray<double> _x(_shape);
  size_t numel = _x.size();
  std::generate_n(_x.data(), numel / 2, [&]() { return uniform(rdv); });
  std::generate_n(_x.data() + numel / 2, numel - numel / 2,
                  [&]() { return uniform_large(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(16);
    rt_config.set_enable_hal_profile(true);
    rt_config.set_enable_pphlo_profile(true);
    rt_config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
    rt_config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto x = infeed<double>(ctx, _x);

    size_t bytes_sent = lctx->GetStats()->sent_bytes;
    auto gelu = hal::intrinsic::nn::f_seg4_silu(ctx, x);
    bytes_sent = lctx->GetStats()->sent_bytes - bytes_sent;

    gelu = hlo::Reveal(ctx, gelu);
    if (lctx->Rank() == 0) {
      return;
    }

    double fxp = std::pow(2., rt_config.fxp_fraction_bits());
    double max_err = 0.;

    DISPATCH_ALL_FIELDS(field, "check", [&]() {
      using sT = std::make_signed<ring2k_t>::type;
      for (int64_t i = 0; i < gelu.numel(); ++i) {
        // x * sigmoid(x)
        double expected = _x[i] * (1. / (1. + std::exp(-_x[i])));
        double got = gelu.data().at<sT>(i) / fxp;
        double e = std::abs(got - expected);
        if (e > max_err) {
          max_err = e;
        }
      }
    });

    printf("seg4_silu max error %f\n", max_err);
  });
}

TEST_P(ActivationTest, NegExp) {
  using namespace spu::mpc;
  FieldType field = std::get<0>(GetParam());
  [[maybe_unused]] bool enable_cheetor = std::get<1>(GetParam());

  Shape shape = {100, 50, 20};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-12.0, 0.0);

  std::vector<size_t> _shape;
  for (int64_t d : shape) {
    _shape.push_back((size_t)d);
  }
  xt::xarray<double> _x(_shape);
  size_t numel = _x.size();
  std::generate_n(_x.data(), numel, [&]() { return uniform(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(field == FM32 ? 10 : 16);
    rt_config.set_enable_hal_profile(true);
    rt_config.set_enable_pphlo_profile(true);
    rt_config.set_fxp_exp_iters(5);
    rt_config.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
    // keep 4bits for approx less
    rt_config.mutable_cheetah_2pc_config()->set_approx_less_precision(4);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);

    auto x = infeed<double>(ctx, _x);

    size_t bytes_sent = lctx->GetStats()->sent_bytes;
    auto nexp = hal::intrinsic::nn::f_neg_exp_taylor(ctx, x);
    bytes_sent = lctx->GetStats()->sent_bytes - bytes_sent;

    nexp = hlo::Reveal(ctx, nexp);
    if (lctx->Rank() == 0) {
      return;
    }

    double fxp = std::pow(2., rt_config.fxp_fraction_bits());
    double max_err = 0.;

    DISPATCH_ALL_FIELDS(field, "check", [&]() {
      using sT = std::make_signed<ring2k_t>::type;
      for (int64_t i = 0; i < nexp.numel(); ++i) {
        double expected = std::exp(_x[i]);
        double got = nexp.data().at<sT>(i) / fxp;

        double e = std::abs(got - expected);
        if (e > max_err) {
          max_err = e;
        }
      }
    });

    printf("nexp max error %f, %f bytes per\n", max_err,
           bytes_sent * 1. / shape.numel());
  });
}

}  // namespace spu::kernel::hal::test
