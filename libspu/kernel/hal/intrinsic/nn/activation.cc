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
#include "libspu/kernel/hal/intrinsic/nn/activation.h"

#include <future>

#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/cheetah/alg.h"
#include "libspu/mpc/cheetah/conversion.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal::intrinsic::nn {

static std::array<Value, 3> ComputeUptoPower4(SPUContext* ctx, const Value& x) {
  SPU_ENFORCE(x.isFxp() and x.isSecret());
  auto x2 = f_square(ctx, x);
  auto x4 = f_square(ctx, x2);
  auto x3 = f_mul(ctx, x, x2);
  return {x2, x3, x4};
}

static std::vector<Value> ComputedBatchLessAP(SPUContext* ctx, const Value& x,
                                              absl::Span<const float> y) {
  std::vector<Value> ret;
  if (not x.isSecret() || ctx->config().protocol() != ProtocolKind::CHEETAH) {
    for (float yy : y) {
      ret.emplace_back(f_less(ctx, x, constant(ctx, yy, x.dtype(), x.shape())));
    }
    return ret;
  }

  KernelEvalContext kctx(ctx);
  auto _ret = spu::mpc::cheetah::BatchLessThan(&kctx, x.data(), y);
  for (auto& d : _ret) {
    d = d.reshape(x.shape());
    ret.emplace_back(d, DT_I1);
  }
  return ret;
}

// sigmoid(x) for x > 0
static Value do_f_sigmoid_positive(SPUContext* ctx, const Value& x) {
  SPU_ENFORCE(x.isFxp() and x.isSecret());
  // a*x^4 + b*x^3 + c*x^2 + d*x + e
  std::array<float, 5> coeffs = {0.49352908920523014, 0.3028510171823098,
                                 -0.06808619212463336, 0.006747908074344234,
                                 -0.0002472776978734074};

  int fxp = ctx->getFxpBits();
  int wk_fxp = std::max(fxp, 21);  // 0.0002472776978734074 ~ 2^{-11}

  auto [x2, x3, x4] = ComputeUptoPower4(ctx, x);

  float scale = 1L << fxp;

  auto seg_3 = _mul(ctx, x3, constant(ctx, coeffs[3], x.dtype(), x.shape()));
  auto seg_2 = _mul(ctx, x2, constant(ctx, coeffs[2], x.dtype(), x.shape()));
  auto seg_1 = _mul(ctx, x, constant(ctx, coeffs[1], x.dtype(), x.shape()));

  auto seg_0 = constant(ctx, coeffs[0] * scale, x.dtype(), x.shape());
  auto sigmoid = _add(ctx, seg_0, seg_1);
  sigmoid = _add(ctx, sigmoid, seg_2);
  sigmoid = _add(ctx, sigmoid, seg_3);

  // NOTE(lwj) take care the x^4 term due to the tiny coefficient
  auto seg_4 = _mul(
      ctx, x4,
      constant(ctx, coeffs[4] * (1L << (wk_fxp - fxp)), x.dtype(), x.shape()));
  if (wk_fxp > fxp) {
    sigmoid = _lshift(ctx, sigmoid, wk_fxp - fxp).setDtype(x.dtype());
  }
  sigmoid = _add(ctx, sigmoid, seg_4);

  sigmoid = _trunc(ctx, sigmoid, wk_fxp).setDtype(x.dtype());

  return sigmoid;
}

static Value do_f_seg3_gelu(SPUContext* ctx, Value x) {
  SPU_ENFORCE(x.isFxp() and x.isSecret());

  // Compute gelu in FM32
  // To prevent overflow in depth-1 mul, we can not use too large fxp
  // We approximate in x \in [-3, 3].
  const int fxp_before = ctx->getFxpBits();
  const int _fxp = 12;
  const int fxp_to_drop = fxp_before - _fxp;
  const float apprx_range = 3.0;

  if (fxp_to_drop > 0) {
    x = _trunc(ctx, x, fxp_to_drop).setDtype(x.dtype());
    const_cast<RuntimeConfig*>(&ctx->config())
        ->set_fxp_fraction_bits(fxp_before - fxp_to_drop);
  }

  const auto ONE = _constant(ctx, 1, x.shape());
  const auto True = _and(ctx, ONE, ONE);

  auto batch_less_than =
      ComputedBatchLessAP(ctx, x, {-apprx_range, 0.0F, apprx_range});
  auto b1 = _xor(ctx, batch_less_than[1], ONE);  // x >= 0.0
  auto b2 = _xor(ctx, batch_less_than[2], ONE);  // x >= 3
  // -3 <= x <= -3.0
  auto b0 = _xor(ctx, _xor(ctx, ONE, batch_less_than[0]), b2);

  // x = b1 ? x : -x
  auto abs_x = _mux(ctx, b1, x, _negate(ctx, x)).setDtype(x.dtype());

  // seg = a*|x|^4 + b*|x|^3 + c*|x|^2 + d*|x| + e + 0.5x
  std::vector<float> coeffs = {0.001620808531841547, -0.03798164612714154,
                               0.5410550166368381, -0.18352506127082727,
                               0.020848611754127593};
  auto [x2, x3, x4] = ComputeUptoPower4(ctx, abs_x);

  float scale = 1L << ctx->getFxpBits();
  auto seg_4 = _mul(ctx, x4, constant(ctx, coeffs[4], x.dtype(), x.shape()));
  auto seg_3 = _mul(ctx, x3, constant(ctx, coeffs[3], x.dtype(), x.shape()));
  auto seg_2 = _mul(ctx, x2, constant(ctx, coeffs[2], x.dtype(), x.shape()));
  auto seg_1 = _mul(ctx, abs_x, constant(ctx, coeffs[1], x.dtype(), x.shape()));
  auto seg_0 = constant(ctx, coeffs[0] * scale, x.dtype(), x.shape());
  auto seg = _trunc(
      ctx,
      _add(ctx, _mul(ctx, x, constant(ctx, 0.5, x.dtype(), x.shape())),
           _add(ctx, seg_0,
                _add(ctx, seg_1, _add(ctx, seg_2, _add(ctx, seg_3, seg_4))))));
  // x > 3
  auto gelu = _mul(ctx, b2, x);
  // -3 <= x  <= 3
  gelu = _add(ctx, gelu, _mul(ctx, b0, seg));
  gelu.setDtype(x.dtype());

  if (fxp_to_drop > 0) {
    const_cast<RuntimeConfig*>(&ctx->config())
        ->set_fxp_fraction_bits(fxp_before);
    gelu = _lshift(ctx, gelu, fxp_to_drop).setDtype(x.dtype());
  }

  return gelu;
}

Value f_seg3_gelu(SPUContext* ctx, const Value& x_) {
  SPU_TRACE_HAL_LEAF(ctx, x_);
  SPU_ENFORCE(ctx->config().protocol() == ProtocolKind::CHEETAH);

  [[maybe_unused]] size_t sent = ctx->lctx()->GetStats()->sent_bytes;

  // NOTE(lwj): We compute the whole seg3_gelu(x) over a smaller 32-bit ring.
  // We first cast down the share of x to the target ring FM32.
  auto src_field = ctx->config().field();
  auto target_field = FieldType::FM32;

  spu::Value x = [&]() {
    if (src_field == target_field) {
      // noting to do
      return x_;
    }

    KernelEvalContext kctx(ctx);
    mpc::cheetah::CastRing ring_change_kernel;

    spu::Value ret(ring_change_kernel.proc(&kctx, x_.data(), target_field),
                   DT_F32);
    // Because the ring_cast operation is not supported by SPU, we need to call
    // Cheetah's CastRing protocol directly.
    // Also, we need mannually modify the default field in RuntimeConfig to FM32
    // NOTE(lwj): dirty hack to change the current field
    const_cast<RuntimeConfig*>(&ctx->config())->set_field(target_field);
    ctx->getState<spu::mpc::Z2kState>()->setField(target_field);

    return ret;
  }();

  auto gelu = do_f_seg3_gelu(ctx, x);

  if (src_field != target_field) {
    // convert the field and fxp back
    const_cast<RuntimeConfig*>(&ctx->config())->set_field(src_field);
    ctx->getState<spu::mpc::Z2kState>()->setField(src_field);

    KernelEvalContext kctx(ctx);
    mpc::cheetah::CastRing ring_change_kernel;
    gelu = Value(ring_change_kernel.proc(&kctx, gelu.data(), src_field,
                                         SignType::Unknown),
                 x_.dtype());
  }

  sent = ctx->lctx()->GetStats()->sent_bytes - sent;
  SPDLOG_INFO("seg3_gelu {} sent {} MiB", gelu.numel(), sent / 1024. / 1024.);
  return gelu;
}

Value do_f_seg4_silu(SPUContext* ctx, const Value& x,
                     absl::Span<const Value> branch_indicators) {
  SPU_ENFORCE(x.isFxp() and x.isSecret());
  SPU_ENFORCE_EQ(branch_indicators.size(), 3U);
  // branch_indicators[0] <=> |x| <= 8
  // branch_indicators[1] <=> x >= 0
  // branch_indicators[2] <=> x >= 8

  // |x| = x >= 0.0 ? x : -x
  auto abs_x = hal::select(ctx, branch_indicators[1], x, f_negate(ctx, x));

  // sigmoid(|x|)
  auto pos_sigmoid = do_f_sigmoid_positive(ctx, abs_x);

  auto ONE = constant(ctx, 1.0, pos_sigmoid.dtype(), pos_sigmoid.shape());
  // 1 - sigmoid(|x|)
  auto neg_sigmoid = f_sub(ctx, ONE, pos_sigmoid);

  auto sigmoid =
      hal::select(ctx, branch_indicators[1], pos_sigmoid, neg_sigmoid);

  // x >= 8
  auto silu = _mul(ctx, branch_indicators[2], ONE);

  // |x| < 8
  silu = _add(ctx, silu, _mul(ctx, branch_indicators[0], sigmoid))
             .setDtype(x.dtype());

  return f_mul(ctx, x, silu);
}

Value f_seg4_silu(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);
  [[maybe_unused]] size_t sent = ctx->lctx()->GetStats()->sent_bytes;

  auto branch_indicators = [&]() {
    auto src_field = ctx->config().field();
    auto target_field = FieldType::FM32;

    KernelEvalContext kctx(ctx);
    mpc::cheetah::CastRing ring_change_kernel;

    Value x32(ring_change_kernel.proc(&kctx, x.data(), target_field), DT_F32);

    const_cast<RuntimeConfig*>(&ctx->config())->set_field(target_field);
    ctx->getState<spu::mpc::Z2kState>()->setField(target_field);

    const auto ONE = _constant(ctx, 1, x32.shape());
    const auto True = _and(ctx, ONE, ONE);
    // Compute branch indicators in FM32
    auto batch_less_than = ComputedBatchLessAP(ctx, x32, {-8.0, 0.0F, 8.0});

    batch_less_than[1] = _xor(ctx, batch_less_than[1], ONE);
    batch_less_than[2] = _xor(ctx, batch_less_than[2], ONE);
    batch_less_than[0] =
        _xor(ctx, _xor(ctx, batch_less_than[0], batch_less_than[2]), ONE);

    for (size_t i : {0, 1, 2}) {
      // cast back to the src_field
      batch_less_than[i] = Value(
          ring_change_kernel.proc(&kctx, batch_less_than[i].data(), src_field),
          DT_I1);
    }

    const_cast<RuntimeConfig*>(&ctx->config())->set_field(src_field);
    ctx->getState<spu::mpc::Z2kState>()->setField(src_field);
    return batch_less_than;
  }();

  auto silu = do_f_seg4_silu(ctx, x, absl::MakeSpan(branch_indicators));

  sent = ctx->lctx()->GetStats()->sent_bytes - sent;
  SPDLOG_INFO("seg4_silu {} sent {} MiB", silu.numel(), sent / 1024. / 1024.);

  return silu;
}

Value f_neg_exp_taylor(SPUContext* ctx, const Value& x) {
  SPU_TRACE_HAL_LEAF(ctx, x);

  int fxp_exp_iters = ctx->config().fxp_exp_iters();
  SPU_ENFORCE(fxp_exp_iters != 0, "fxp_exp_iters should not be {}",
              fxp_exp_iters);

  [[maybe_unused]] size_t sent = ctx->lctx()->GetStats()->sent_bytes;
  const auto ONE = _constant(ctx, 1, x.shape());
  const auto True = _and(ctx, ONE, ONE);
  float neg_range = -14.0;
  auto is_not_too_small =
      _xor(ctx, True, ComputedBatchLessAP(ctx, x, {neg_range})[0]);

  // 1 + x/2^n
  Value res = f_add(ctx, _trunc(ctx, x, fxp_exp_iters).setDtype(x.dtype()),
                    constant(ctx, 1.0F, x.dtype(), x.shape()));

  // (1 + x/2^n)^(2^n)
  for (int i = 0; i < fxp_exp_iters; i++) {
    res = f_square(ctx, res);
  }

  // convert the field and fxp back
  auto ret = _mul(ctx, is_not_too_small, res).setDtype(x.dtype());

  sent = ctx->lctx()->GetStats()->sent_bytes - sent;
  SPDLOG_INFO("f_nexp {} sent {} MiB", x.numel(), sent / 1024. / 1024.);
  return ret;
}

}  // namespace spu::kernel::hal::intrinsic::nn
