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

#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/core/value.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/prot_wrapper.h"
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

Value f_seg3_gelu(SPUContext* ctx, const Value& x_) {
  SPU_ENFORCE(x_.isFxp());
  SPU_TRACE_HAL_LEAF(ctx, x_);

  SPU_ENFORCE(ctx->config().protocol() == ProtocolKind::CHEETAH);
  SPU_ENFORCE(x_.isFxp() and x_.isSecret());

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

  // Compute gelu in FM32
  // To prevent overflow in depth-1 mul, we can not use too large fxp
  // We approximate in x \in [-3, 3].
  const int fxp_before = ctx->getFxpBits();
  const int _fxp = 12;
  const int fxp_to_drop = fxp_before - _fxp;
  const float apprx_range = 3.0;

  if (fxp_to_drop > 0) {
    x = _trunc(ctx, x, fxp_to_drop).setDtype(x_.dtype());
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

  if (src_field != target_field) {
    // convert the field and fxp back
    const_cast<RuntimeConfig*>(&ctx->config())->set_field(src_field);
    const_cast<RuntimeConfig*>(&ctx->config())
        ->set_fxp_fraction_bits(fxp_before);
    ctx->getState<spu::mpc::Z2kState>()->setField(src_field);

    KernelEvalContext kctx(ctx);
    mpc::cheetah::CastRing ring_change_kernel;
    gelu = Value(ring_change_kernel.proc(&kctx, gelu.data(), src_field,
                                         SignType::Unknown),
                 x_.dtype());
  }

  if (fxp_to_drop > 0) {
    gelu = _lshift(ctx, gelu, fxp_to_drop).setDtype(x_.dtype());
  }

  sent = ctx->lctx()->GetStats()->sent_bytes - sent;
  SPDLOG_INFO("seg3_gelu {} sent {} MiB", gelu.numel(), sent / 1024. / 1024.);
  return gelu;
}

Value f_seg4_silu(SPUContext* ctx, const Value& x) {
  SPU_ENFORCE(x.isFxp());
  SPU_TRACE_HAL_LEAF(ctx, x);
  const auto ONE = _constant(ctx, 1, x.shape());
  const auto True = _and(ctx, ONE, ONE);

  auto batch_less_than = ComputedBatchLessAP(ctx, x, {-8.0F, -4.0F, 4.0F});
  const auto& b1 = batch_less_than[1];           // x < -4.0
  const auto& b0 = batch_less_than[0];           // x < -8.0
  auto b2 = _xor(ctx, batch_less_than[2], ONE);  // x >= 4.0
  auto b3 = _xor(ctx, True, _xor(ctx, b1, b2));  // x in  [-4.0, 4.0)
  auto b4 = _xor(ctx, b0, b1);                   // x in [-8, -4.0)

  // seg1 = a[2] * x^2 + a[1] * x + a[0]
  // seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
  std::vector<float> a_coeffs = {-0.3067541139982155, -0.0819767021525476,
                                 -0.0055465625580307};
  std::vector<float> b_coeffs = {
      0.0085064025895951, 0.5, 0.2281430841728270, 0.0,
      -0.011113046708173, 0.0, 0.0002743776353465};

  const float scale = 1L << ctx->getFxpBits();
  auto [x2, x3, x4] = ComputeUptoPower4(ctx, x);
  auto x6 = f_square(ctx, x3);

  auto seg1_2 = _mul(ctx, x2, constant(ctx, a_coeffs[2], x.dtype(), x.shape()));
  auto seg1_1 = _mul(ctx, x, constant(ctx, a_coeffs[1], x.dtype(), x.shape()));
  auto seg1_0 = constant(ctx, a_coeffs[0] * scale, x.dtype(), x.shape());
  auto seg2_6 = _mul(ctx, x6, constant(ctx, b_coeffs[6], x.dtype(), x.shape()));
  auto seg2_4 = _mul(ctx, x4, constant(ctx, b_coeffs[4], x.dtype(), x.shape()));
  auto seg2_2 = _mul(ctx, x2, constant(ctx, b_coeffs[2], x.dtype(), x.shape()));
  auto seg2_1 = _mul(ctx, x, constant(ctx, b_coeffs[1], x.dtype(), x.shape()));
  auto seg2_0 = constant(ctx, b_coeffs[0] * scale, x.dtype(), x.shape());
  auto seg2 =
      _trunc(ctx, _add(ctx, seg2_0,
                       _add(ctx, seg2_1,
                            _add(ctx, seg2_2, _add(ctx, seg2_4, seg2_6)))))
          .setDtype(x.dtype());
  auto seg1 = _trunc(ctx, _add(ctx, seg1_0, _add(ctx, seg1_1, seg1_2)))
                  .setDtype(x.dtype());
  auto ret = _mul(ctx, b2, x);
  ret = _add(ctx, ret, _mul(ctx, b4, seg1));
  ret = _add(ctx, ret, _mul(ctx, b3, seg2));
  return ret.setDtype(x.dtype());
  return _mul(ctx, b4, seg1).setDtype(x.dtype());
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
