// Copyright 2021 Ant Group Co., Ltd.
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
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/bumblebee/kernels/algo.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::kernel::hal::intrinsic::nn {

static std::array<Value, 3> ComputeUptoPower4(SPUContext* ctx, const Value& x) {
  // if (not x.isSecret() || ctx->config().protocol() !=
  // ProtocolKind::BUMBLEBEE) {
  auto x2 = f_square(ctx, x);
  auto x4 = f_square(ctx, x2);
  auto x3 = f_mul(ctx, x, x2);
  return {x2, x3, x4};
  // }
  //
  // KernelEvalContext kctx(ctx);
  // auto [_x2, _x3, _x4] =
  //     spu::mpc::bumblebee::ComputeUptoPower4(&kctx, x.data());
  // return {Value(_x2, x.dtype()), Value(_x3, x.dtype()), Value(_x4,
  // x.dtype())};
}

static std::vector<Value> ComputedBatchLessAP(SPUContext* ctx, const Value& x,
                                              absl::Span<const float> y) {
  std::vector<Value> ret;
  if (not x.isSecret() || ctx->config().protocol() != ProtocolKind::BUMBLEBEE) {
    for (float yy : y) {
      ret.emplace_back(f_less(ctx, x, constant(ctx, yy, x.dtype(), x.shape())));
    }
    return ret;
  }

  KernelEvalContext kctx(ctx);
  auto _ret = spu::mpc::bumblebee::BatchLessThan(&kctx, x.data(), y);
  for (auto& d : _ret) {
    d = d.reshape(x.shape());
    ret.emplace_back(d, DT_I1);
  }
  return ret;
}

static Value f_gelu_bumblebee(SPUContext* ctx, const Value& _x) {
  [[maybe_unused]] size_t sent = ctx->lctx()->GetStats()->sent_bytes;
  auto src_field = ctx->config().field();
  auto target_field = FieldType::FM32;

  KernelEvalContext kctx(ctx);
  Value x(spu::mpc::bumblebee::ChangeRing(&kctx, _x.data(), target_field),
          _x.dtype());

  // Compute gelu in FM32
  // To prevent overflow in depth-1 mul, we can not use too large fxp
  // We approximate in x \in [-3, 3].
  const int fxp_before = ctx->getFxpBits();
  const int fxp_to_drop = fxp_before - ((SizeOf(target_field) * 8 - 8) / 2);

  const_cast<RuntimeConfig*>(&ctx->config())->set_field(target_field);
  ctx->getState<spu::mpc::Z2kState>()->setField(target_field);
  if (fxp_to_drop > 0) {
    x = _trunc(ctx, x, fxp_to_drop).setDtype(_x.dtype());
    const_cast<RuntimeConfig*>(&ctx->config())
        ->set_fxp_fraction_bits(fxp_before - fxp_to_drop);
  }

  const auto ONE = _constant(ctx, 1, x.shape());
  const auto True = _and(ctx, ONE, ONE);

  auto batch_less_than = ComputedBatchLessAP(ctx, x, {-3.0F, 0.0F, 3.0F});
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

  float scale = 1L << ctx->getFxpBits();
  auto [x2, x3, x4] = ComputeUptoPower4(ctx, abs_x);
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

  // convert back
  const_cast<RuntimeConfig*>(&ctx->config())->set_field(src_field);
  const_cast<RuntimeConfig*>(&ctx->config())->set_fxp_fraction_bits(fxp_before);
  ctx->getState<spu::mpc::Z2kState>()->setField(src_field);

  gelu = Value(spu::mpc::bumblebee::ChangeRing(&kctx, gelu.data(), src_field,
                                               SignType::Unknown),
               _x.dtype());
  if (fxp_to_drop > 0) {
    gelu = _lshift(ctx, gelu, fxp_to_drop).setDtype(_x.dtype());
  }

  sent = ctx->lctx()->GetStats()->sent_bytes - sent;
  SPDLOG_INFO("gelu {} sent {} MiB", gelu.numel(), sent / 1024. / 1024.);
  return gelu;
}

Value f_gelu(SPUContext* ctx, const Value& x) {
  SPU_ENFORCE(x.isFxp());
  SPU_TRACE_HAL_LEAF(ctx, x);

  if (x.isSecret() && ctx->config().protocol() == ProtocolKind::BUMBLEBEE) {
    return f_gelu_bumblebee(ctx, x);
  }

  const auto ONE = _constant(ctx, 1, x.shape());
  const auto True = _and(ctx, ONE, ONE);

  auto batch_less_than = ComputedBatchLessAP(ctx, x, {-4.0F, -1.95F, 3.0F});
  const auto& b1 = batch_less_than[1];           // x < -1.95
  const auto& b0 = batch_less_than[0];           // x < -4.0
  auto b2 = _xor(ctx, batch_less_than[2], ONE);  // x >= 3
  auto b3 = _xor(ctx, True, _xor(ctx, b1, b2));  // x in  [-1.95, 3)
  auto b4 = _xor(ctx, b0, b1);                   // x in [-4, -1.95)

  // seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
  // seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
  std::vector<float> a_coeffs = {
      -0.5054031199708174,
      -0.42226581151983866,
      -0.11807612951181953,
      -0.011034134030615728,
  };
  std::vector<float> b_coeffs = {
      0.008526321541038084,  0.5, 0.3603292692789629,    0.0,
      -0.037688200365904236, 0.0, 0.0018067462606141187,
  };
  const float scale = 1L << ctx->getFxpBits();
  auto [x2, x3, x4] = ComputeUptoPower4(ctx, x);
  auto x6 = f_square(ctx, x3);
  auto seg1_3 = _mul(ctx, x3, constant(ctx, a_coeffs[3], x.dtype(), x.shape()));
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
  auto seg1 = _trunc(ctx, _add(ctx, seg1_0,
                               _add(ctx, seg1_1, _add(ctx, seg1_2, seg1_3))))
                  .setDtype(x.dtype());
  auto gelu = _mul(ctx, b2, x);
  gelu = _add(ctx, gelu, _mul(ctx, b4, seg1));
  gelu = _add(ctx, gelu, _mul(ctx, b3, seg2));
  gelu.setDtype(x.dtype());
  return gelu;
}

Value f_silu(SPUContext* ctx, const Value& x) {
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
}  // namespace spu::kernel::hal::intrinsic::nn
