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

#include "libspu/mpc/cheetah/kernels/compare.h"

#include <future>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/env.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
// Math:
//  msb(x0 + x1 mod 2^k) = msb(x0) ^ msb(x1) ^ 1{(x0 + x1) > 2^{k-1} - 1}
//  The carry bit
//     1{(x0 + x1) > 2^{k - 1} - 1} = 1{x0 > 2^{k - 1} - 1 - x1}
//  is computed using a Millionare protocol.
NdArrayRef MsbA2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  const size_t shft = nbits - 1;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return out.as(makeType<BShrTy>(field, 1));
  }

  const int64_t num_job = InitOTState(ctx, numel);
  const int64_t work_load = num_job == 0 ? 0 : CeilDiv(numel, num_job);
  const int rank = ctx->getState<Communicator>()->getRank();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    NdArrayRef adjusted = ring_zeros(field, {numel});
    auto xinp = NdArrayView<const u2k>(x);
    auto xadj = NdArrayView<u2k>(adjusted);

    if (rank == 0) {
      // x0
      pforeach(0, numel, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      // 2^{k - 1} - 1 - x1
      pforeach(0, numel, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    NdArrayRef carry_bit(x.eltype(), x.shape());
    TiledDispatch(ctx, num_job, [&](int64_t job) {
      int64_t slice_bgn = std::min(job * work_load, numel);
      int64_t slice_end = std::min(slice_bgn + work_load, numel);

      if (slice_end == slice_bgn) {
        return;
      }

      CompareProtocol prot(ctx->getState<CheetahOTState>()->get(job));

      // 1{x0 > 2^{k - 1} - 1 - x1}
      auto out_slice =
          prot.Compute(adjusted.slice({slice_bgn}, {slice_end}, {1}),
                       /*greater*/ true, nbits);

      std::memcpy(&carry_bit.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    });

    // [msb(x)]_B <- [1{x0 + x1 > 2^{k- 1} - 1]_B ^ msb(x0)
    NdArrayView<u2k> _carry_bit(carry_bit);
    pforeach(0, numel, [&](int64_t i) { _carry_bit[i] ^= (xinp[i] >> shft); });

    return carry_bit.as(makeType<BShrTy>(field, 1));
  });
}

NdArrayRef LessAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        const NdArrayRef& y) const {
  SPU_TRACE_ACTION(GET_TRACER(ctx), ctx->lctx(), (TR_MPC | TR_LAR), (~TR_MPC),
                   "less_ap");
  int64_t n = x.numel();
  if (n == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  const int rank = ctx->getState<Communicator>()->getRank();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  size_t fxp = ctx->sctx()->config().fxp_fraction_bits();
  bool allow_approx_less_than =
      TestEnvFlag(EnvFlag::SPU_CHEETAH_ENABLE_APPROX_LESS_THAN);

  size_t bits_skip = allow_approx_less_than ? fxp - 4 : 0;

  // msb((x - y) / 2^d) using
  auto fx = x.reshape({n});
  auto fy = y.reshape({n});
  NdArrayRef minus(x.eltype(), {n});
  if (rank == 0) {
    minus = ring_sub(fx, fy);
  } else {
    minus = fx.clone();
  }

  if (bits_skip > 0) {
    ring_arshift_(minus, bits_skip);
  }

  MsbA2B msb_prot(SizeOf(field) * 8 - bits_skip);
  return msb_prot.proc(ctx, minus).reshape(x.shape());
}

NdArrayRef LessPA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        const NdArrayRef& y) const {
  SPU_TRACE_ACTION(GET_TRACER(ctx), ctx->lctx(), (TR_MPC | TR_LAR), (~TR_MPC),
                   "less_pa");
  int64_t n = x.numel();
  if (n == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  const int rank = ctx->getState<Communicator>()->getRank();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  size_t fxp = ctx->sctx()->config().fxp_fraction_bits();
  bool allow_approx_less_than =
      TestEnvFlag(EnvFlag::SPU_CHEETAH_ENABLE_APPROX_LESS_THAN);
  size_t bits_skip = allow_approx_less_than ? fxp - 4 : 0;
  // msb((x - y) / 2^d) using
  auto fx = x.reshape({n});
  auto fy = y.reshape({n});
  NdArrayRef minus(x.eltype(), {n});
  if (rank == 0) {
    minus = ring_sub(fx, fy);
  } else {
    minus = ring_neg(fy);
  }

  if (bits_skip > 0) {
    ring_arshift_(minus, bits_skip);
  }

  MsbA2B msb_prot(SizeOf(field) * 8 - bits_skip);
  return msb_prot.proc(ctx, minus).reshape(x.shape());
}

NdArrayRef EqualAP::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  SPU_TRACE_ACTION(GET_TRACER(ctx), ctx->lctx(), (TR_MPC | TR_LAR), (~TR_MPC),
                   "equal_ap");
  int bits = TestEnvInt(EnvFlag::SPU_CHEETAH_SET_IEQUAL_BITS);
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  bits = std::max(0, std::min<int>(bits, SizeOf(field) * 8));

  EqualAA equal_aa(bits);
  // TODO(juhou): Can we use any place holder to indicate the dummy 0s.
  if (0 == ctx->getState<Communicator>()->getRank()) {
    return equal_aa.proc(ctx, x, ring_zeros(field, x.shape()));
  } else {
    return equal_aa.proc(ctx, x, y);
  }
}

NdArrayRef EqualAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  const int64_t numel = x.numel();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t nbits = nbits_ == 0 ? SizeOf(field) * 8 : nbits_;
  SPU_ENFORCE(nbits <= 8 * SizeOf(field));

  NdArrayRef eq_bit(x.eltype(), x.shape());
  if (numel == 0) {
    return eq_bit.as(makeType<BShrTy>(field, 1));
  }

  const int64_t num_job = InitOTState(ctx, numel);
  const int64_t work_load = num_job == 0 ? 0 : CeilDiv(numel, num_job);
  const int rank = ctx->getState<Communicator>()->getRank();

  //     x0 + x1 = y0 + y1 mod 2k
  // <=> x0 - y0 = y1 - x1 mod 2k
  NdArrayRef adjusted;
  if (rank == 0) {
    adjusted = ring_sub(x, y);
  } else {
    adjusted = ring_sub(y, x);
  }

  // Need 1D array
  adjusted = adjusted.reshape({adjusted.numel()});
  TiledDispatch(ctx, num_job, [&](int64_t job) {
    int64_t slice_bgn = std::min(job * work_load, numel);
    int64_t slice_end = std::min(slice_bgn + work_load, numel);

    if (slice_end == slice_bgn) {
      return;
    }

    EqualProtocol prot(ctx->getState<CheetahOTState>()->get(job));
    auto out_slice =
        prot.Compute(adjusted.slice({slice_bgn}, {slice_end}, {1}), nbits);

    std::memcpy(&eq_bit.at(slice_bgn), &out_slice.at(0),
                out_slice.numel() * out_slice.elsize());
  });

  return eq_bit.as(makeType<BShrTy>(field, 1));
}

}  // namespace spu::mpc::cheetah