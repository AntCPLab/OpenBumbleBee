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
#include "libspu/mpc/cheetah/kernels/matmul.h"

#include <future>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
// A is (M, K); B is (K, N)
NdArrayRef MatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms homomorphically
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->DotOLE(x, dupx.get(), dim3, true);
    } else {
      return dot_prot->DotOLE(y, dupx.get(), dim3, false);
    }
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->DotOLE(y, conn, dim3, false);
  } else {
    x1y0 = dot_prot->DotOLE(x, conn, dim3, true);
  }

  auto ret = ring_mmul(x, y);
  ring_add_(ret, x1y0);
  return ring_add(ret, task.get()).as(x.eltype());
}

NdArrayRef MatMulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();

  if (owner == rank) {
    auto field = ptype->field();
    DISPATCH_ALL_FIELDS(field, "check_binary", [&]() {
      NdArrayView<ring2k_t> _y(y);
      bool is_binary = true;
      for (int64_t i = 0; i < _y.numel(); ++i) {
        if (_y[i] > 1) {
          is_binary = false;
          break;
        }
      }

      SPDLOG_INFO("MatMulAV: is RHS binary {}", is_binary);
    });
  }

  NdArrayRef out;
  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  // (x0 + x1)*y = <x0 * y>_0 + <x0 * y>_1 + x1 * y
  if (rank == owner) {
    // Compute <y * x0>
    auto he_task = std::async(
        [&]() -> spu::NdArrayRef { return dot_prot->DotOLE(y, dim3, false); });
    out = ring_mmul(x, y);
    ring_add_(out, he_task.get());
  } else {
    out = dot_prot->DotOLE(x, dim3, true);
  }
  return out.as(x.eltype());
}

NdArrayRef MatMulVVS::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                           const NdArrayRef& y) const {
  auto out_type = makeType<cheetah::AShrTy>(ctx->sctx()->getField());
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(out_type, {x.shape()[0], y.shape()[1]});
  }
  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();

  const int self_rank = comm->getRank();
  auto lhs_owner = x.eltype().as<Priv2kTy>()->owner();

  const Shape3D dim3 = {x.shape()[0], x.shape()[1], y.shape()[1]};
  if (self_rank == lhs_owner) {
    return dot_prot->DotOLE(x, dim3, /*is_lhs*/ true).as(out_type);
  } else {
    return dot_prot->DotOLE(y, dim3, /*is_lhs*/ false).as(out_type);
  }
}

void BatchMatMulAA::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  auto xs = lhs.shape();
  auto ys = rhs.shape();
  SPU_ENFORCE(xs.ndim() == ys.ndim(), "ndim mismatch: lhs={}, rhs={}", xs, ys);
  SPU_ENFORCE(xs[0] == ys[0], "batch mismatch: lhs={}, rhs={}", xs, ys);
  SPU_ENFORCE(xs[2] == ys[1], "shape mismatch: lhs={}, rhs={}", xs, ys);
  ctx->setOutput(WrapValue(proc(ctx, lhs.data(), rhs.data())));
}

// A is (B, M, K); B is (B, K, N)
NdArrayRef BatchMatMulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }
  SPU_ENFORCE(x.ndim() == 3 && y.ndim() == 3);
  SPU_ENFORCE_EQ(x.shape()[0], y.shape()[0]);
  SPU_ENFORCE_EQ(x.shape()[2], y.shape()[1]);

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();

  // (x0 + x1) * (y0 + y1)
  // Compute the cross terms
  const Shape4D dim4 = {x.shape()[0], x.shape()[1], x.shape()[2], y.shape()[2]};

  auto* conn = comm->lctx().get();
  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    // Compute x0*y1
    if (rank == 0) {
      return dot_prot->BatchDotOLE(x, dupx.get(), dim4, true);
    } else {
      return dot_prot->BatchDotOLE(y, dupx.get(), dim4, false);
    }
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = dot_prot->BatchDotOLE(y, conn, dim4, false);
  } else {
    x1y0 = dot_prot->BatchDotOLE(x, conn, dim4, true);
  }

  const Strides strides(x.shape().size(), 1);
  Index lhs_slice_end(x.shape().begin(), x.shape().end());
  Index rhs_slice_end(y.shape().begin(), y.shape().end());
  Index lhs_slice_begin(3, 0);
  Index rhs_slice_begin(3, 0);
  NdArrayRef out(x.eltype(), {dim4[0], dim4[1], dim4[3]});
  for (int64_t b = 0; b < dim4[0]; ++b) {
    lhs_slice_begin[0] = b;
    lhs_slice_end[0] = b + 1;
    rhs_slice_begin[0] = b;
    rhs_slice_end[0] = b + 1;
    auto lhs = x.slice(lhs_slice_begin, lhs_slice_end, strides)
                   .reshape({dim4[1], dim4[2]});
    auto rhs = y.slice(rhs_slice_begin, rhs_slice_end, strides)
                   .reshape({dim4[2], dim4[3]});

    auto out_slice = out.slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, strides);
    out_slice = out_slice.reshape({dim4[1], dim4[3]});
    ring_mmul_(out_slice, lhs, rhs);
  }

  ring_add_(out, x1y0);
  ring_add_(out, task.get());
  return out.as(x.eltype());
}

void BatchMatMulAV::evaluate(KernelEvalContext* ctx) const {
  const auto& lhs = ctx->getParam<Value>(0);
  const auto& rhs = ctx->getParam<Value>(1);
  auto xs = lhs.shape();
  auto ys = rhs.shape();
  SPU_ENFORCE(xs.ndim() == ys.ndim(), "ndim mismatch: lhs={}, rhs={}", xs, ys);
  SPU_ENFORCE(xs[0] == ys[0], "batch mismatch: lhs={}, rhs={}", xs, ys);
  SPU_ENFORCE(xs[2] == ys[1], "shape mismatch: lhs={}, rhs={}", xs, ys);
  ctx->setOutput(WrapValue(proc(ctx, lhs.data(), rhs.data())));
}

// A is (B, M, K); B is (B, K, N)
NdArrayRef BatchMatMulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                               const NdArrayRef& y) const {
  if (0 == x.numel() || 0 == y.numel()) {
    return NdArrayRef(x.eltype(), {x.shape()[0], y.shape()[1]});
  }
  SPU_ENFORCE(x.ndim() == 3 && y.ndim() == 3);
  SPU_ENFORCE_EQ(x.shape()[0], y.shape()[0]);
  SPU_ENFORCE_EQ(x.shape()[2], y.shape()[1]);

  auto* comm = ctx->getState<Communicator>();
  auto* dot_prot = ctx->getState<CheetahDotState>()->get();
  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();

  // (x0 + x1)*y = <x0 * y>_0 + <x0 * y>_1 + x1 * y
  const Shape4D dim4 = {x.shape()[0], x.shape()[1], x.shape()[2], y.shape()[2]};

  NdArrayRef out;
  if (rank != owner) {
    out = dot_prot->BatchDotOLE(x, comm->lctx().get(), dim4, true);
  } else {
    out = dot_prot->BatchDotOLE(y, comm->lctx().get(), dim4, false);

    const Strides strides(x.shape().size(), 1);
    Index lhs_slice_end(x.shape().begin(), x.shape().end());
    Index rhs_slice_end(y.shape().begin(), y.shape().end());
    Index lhs_slice_begin(3, 0);
    Index rhs_slice_begin(3, 0);

    for (int64_t b = 0; b < dim4[0]; ++b) {
      lhs_slice_begin[0] = b;
      lhs_slice_end[0] = b + 1;
      rhs_slice_begin[0] = b;
      rhs_slice_end[0] = b + 1;
      auto lhs = x.slice(lhs_slice_begin, lhs_slice_end, strides)
                     .reshape({dim4[1], dim4[2]});
      auto rhs = y.slice(rhs_slice_begin, rhs_slice_end, strides)
                     .reshape({dim4[2], dim4[3]});
      auto local = ring_mmul(lhs, rhs);

      auto out_slice = out.slice({b, 0, 0}, {b + 1, dim4[1], dim4[3]}, strides);
      out_slice = out_slice.reshape({dim4[1], dim4[3]});
      ring_add_(out_slice, local);
    }
  }

  return out.as(x.eltype());
}

NdArrayRef MatMulAP::proc(KernelEvalContext*, const NdArrayRef& x,
                          const NdArrayRef& y) const {
  return ring_mmul(x, y).as(x.eltype());
}

}  // namespace spu::mpc::cheetah
