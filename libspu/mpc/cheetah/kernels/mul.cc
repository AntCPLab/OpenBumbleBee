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
#include "libspu/mpc/cheetah/kernels/mul.h"

#include <future>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/env.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

/* #include "libspu/mpc/cheetah/ot/util.h" */
NdArrayRef OpenShare(const NdArrayRef& shr, ReduceOp op,
                     std::shared_ptr<Communicator> comm, size_t nbits = 0);

NdArrayRef MulA1B::proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                        const NdArrayRef& bshr) const {
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();
  NdArrayRef out(ashr.eltype(), ashr.shape());

  if (numel == 0) {
    return out;
  }

  const int64_t nworker = InitOTState(ctx, numel);
  const int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  // Need 1D Array
  auto flatten_a = ashr.reshape({ashr.numel()});
  auto flatten_b = bshr.reshape({bshr.numel()});
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(job * work_load, numel);
      int64_t slice_end = std::min(slice_bgn + work_load, numel);

      if (slice_end == slice_bgn) {
        break;
      }

      auto out_slice = ctx->getState<CheetahOTState>()->get(job)->Multiplexer(
          flatten_a.slice({slice_bgn}, {slice_end}, {1}),
          flatten_b.slice({slice_bgn}, {slice_end}, {1}));

      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

NdArrayRef MulA1BV::proc(KernelEvalContext* ctx, const NdArrayRef& ashr,
                         const NdArrayRef& bshr) const {
  SPU_TRACE_MPC_LEAF(ctx, "mul_a1bv");
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  SPU_ENFORCE_EQ(ashr.shape(), bshr.shape());
  const int64_t numel = ashr.numel();
  const auto* ptype = bshr.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");

  const int owner = ptype->owner();

  NdArrayRef out(ashr.eltype(), ashr.shape());
  if (numel == 0) {
    return out;
  }

  const int64_t nworker = InitOTState(ctx, numel);
  const int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  // Need 1D Array
  auto flatten_a = ashr.reshape({ashr.numel()});
  // NOTE(lwj): private type might not being boolean
  auto flatten_b =
      bshr.reshape({bshr.numel()}).as(makeType<BShrTy>(ptype->field(), 1));
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(job * work_load, numel);
      int64_t slice_end = std::min(slice_bgn + work_load, numel);

      if (slice_end == slice_bgn) {
        break;
      }

      NdArrayRef out_slice;
      if (rank == owner) {
        out_slice =
            ctx->getState<CheetahOTState>()->get(job)->MultiplexerOnPrivate(
                flatten_a.slice({slice_bgn}, {slice_end}, {1}),
                flatten_b.slice({slice_bgn}, {slice_end}, {1}));
      } else {
        out_slice =
            ctx->getState<CheetahOTState>()->get(job)->MultiplexerOnPrivate(
                flatten_a.slice({slice_bgn}, {slice_end}, {1}));
      }

      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.numel() * out_slice.elsize());
    }
  });

  return out;
}

NdArrayRef MulAA::squareDirectly(KernelEvalContext* ctx,
                                 const NdArrayRef& x) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  //   (x0 + x1) * (x0 + x1)
  // = x0^2 + 2*<x0*x1> + x1^2
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const int rank = comm->getRank();

  auto fx = x.reshape({numel});
  int64_t nhalf = numel <= 8192 ? numel : numel / 2;

  auto subtask = std::async([&]() -> spu::NdArrayRef {
    return mul_prot->MulOLE(fx.slice({0}, {nhalf}, {1}), rank == 0);
  });

  NdArrayRef mul1;
  if (nhalf < numel) {
    auto dupx = ctx->getState<CheetahMulState>()->duplx();
    mul1 = mul_prot->MulOLE(fx.slice({nhalf}, {numel}, {1}), dupx.get(),
                            rank == 1);
  }
  auto mul0 = subtask.get();

  NdArrayRef x0x1(x.eltype(), {numel});
  std::memcpy(&x0x1.at(0), &mul0.at(0), mul0.elsize() * nhalf);
  if (nhalf < numel) {
    std::memcpy(&x0x1.at(nhalf), &mul1.at(0), mul1.elsize() * mul1.numel());
  }
  ring_add_(x0x1, x0x1);
  x0x1 = x0x1.reshape(x.shape());

  return ring_add(x0x1, ring_mul(x, x)).as(x.eltype());
}

NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  auto* mul_prot = ctx->getState<CheetahMulState>()->get()->getImpl();
  mul_prot->Initialize(ctx->sctx()->getField());

  int64_t batch_sze = mul_prot->OLEBatchSize();
  int64_t numel = x.numel();

  if (x.data() == y.data() && x.strides() == y.strides() &&
      4 * x.numel() >= batch_sze) {
    return squareDirectly(ctx, x);
  }

  if (numel >= batch_sze) {
    return mulDirectly(ctx, x, y);
  }
  return mulWithBeaver(ctx, x, y);
}

NdArrayRef MulAA::mulWithBeaver(KernelEvalContext* ctx, const NdArrayRef& x,
                                const NdArrayRef& y) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  auto [a, b, c] =
      ctx->getState<CheetahMulState>()->TakeCachedBeaver(field, numel);
  YACL_ENFORCE_EQ(a.numel(), numel);

  a = a.reshape(x.shape());
  b = b.reshape(x.shape());
  c = c.reshape(x.shape());

  auto* comm = ctx->getState<Communicator>();
  // Open x - a & y - b
  auto res = vmap({ring_sub(x, a), ring_sub(y, b)}, [&](const NdArrayRef& s) {
    return comm->allReduce(ReduceOp::ADD, s, kBindName);
  });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_mul(x_a, b), ring_mul(y_b, a));
  ring_add_(z, c);

  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(x.eltype());
}

NdArrayRef MulAA::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  // (x0 + x1) * (y0+ y1)
  // Compute the cross terms x0*y1, x1*y0 homomorphically
  auto* comm = ctx->getState<Communicator>();
  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  const int rank = comm->getRank();
  auto fx = x.reshape({x.numel()});
  auto fy = y.reshape({y.numel()});

  auto dupx = ctx->getState<CheetahMulState>()->duplx();
  std::future<NdArrayRef> task = std::async(std::launch::async, [&] {
    if (rank == 0) {
      return mul_prot->MulOLE(fx, dupx.get(), true);
    }
    return mul_prot->MulOLE(fy, dupx.get(), false);
  });

  NdArrayRef x1y0;
  if (rank == 0) {
    x1y0 = mul_prot->MulOLE(fy, false);
  } else {
    x1y0 = mul_prot->MulOLE(fx, true);
  }

  x1y0 = x1y0.reshape(x.shape());
  NdArrayRef x0y1 = task.get().reshape(x.shape());
  return ring_add(x0y1, ring_add(x1y0, ring_mul(x, y))).as(x.eltype());
}

NdArrayRef MulAV::mulDirectly(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) const {
  const int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }
  auto* comm = ctx->getState<Communicator>();
  const int rank = comm->getRank();
  const auto* ptype = y.eltype().as<Priv2kTy>();
  SPU_ENFORCE(ptype != nullptr, "rhs should be a private type");
  const int owner = ptype->owner();

  if (rank != owner) {
    auto is_binary = comm->recv<uint8_t>(comm->nextRank(), "sync_type");
    if (is_binary[0] == 1) {
      MulA1BV mul_a1bv;
      return mul_a1bv.proc(ctx, x, y);
    }
  } else {
    auto field = ptype->field();

    uint8_t is_binary = 1;
    DISPATCH_ALL_FIELDS(field, "check", [&]() {
      NdArrayView<const ring2k_t> _y(y);
      for (int64_t i = 0; i < y.numel(); ++i) {
        if (_y[i] > 1) {
          is_binary = 0;
          break;
        }
      }
    });

    comm->sendAsync<uint8_t>(comm->nextRank(), {&is_binary, 1}, "sync_type");
    if (is_binary == 1) {
      MulA1BV mul_a1bv;
      return mul_a1bv.proc(ctx, x, y);
    }
  }

  auto* mul_prot = ctx->getState<CheetahMulState>()->get();
  // (x0 * x1) * y
  // <x0 * y> + x1 * y
  auto fx = x.reshape({numel});
  NdArrayRef out;

  // compute <x0 * y>
  if (rank != owner) {
    out = mul_prot->MulOLE(fx, /*eval*/ true);
  } else {
    auto fy = y.reshape({numel});
    out = mul_prot->MulOLE(fy, /*eval*/ false);
    ring_add_(out, ring_mul(fx, fy));
  }

  return out.reshape(x.shape()).as(x.eltype());
}

NdArrayRef MulAV::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                       const NdArrayRef& y) const {
  SPU_ENFORCE_EQ(x.shape(), y.shape());
  return mulDirectly(ctx, x, y);
}

NdArrayRef MulAP::proc(KernelEvalContext*, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  return ring_mul(lhs, rhs).as(lhs.eltype());
}
}  // namespace spu::mpc::cheetah
