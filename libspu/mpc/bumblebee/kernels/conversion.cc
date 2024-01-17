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
#include "libspu/mpc/bumblebee/kernels/conversion.h"

#include "yacl/utils/parallel.h"

#include "libspu/core/trace.h"
#include "libspu/mpc/ab_api.h"
#include "libspu/mpc/bumblebee/state.h"
#include "libspu/mpc/bumblebee/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::bumblebee {

namespace {
size_t getNumBits(const NdArrayRef& in) {
  if (in.eltype().isa<Pub2kTy>()) {
    const auto field = in.eltype().as<Pub2kTy>()->field();
    return DISPATCH_ALL_FIELDS(field, "_",
                               [&]() { return maxBitWidth<ring2k_t>(in); });
  } else if (in.eltype().isa<BShrTy>()) {
    return in.eltype().as<BShrTy>()->nbits();
  } else {
    SPU_THROW("should not be here, {}", in.eltype());
  }
}

NdArrayRef makeBShare(const NdArrayRef& r, FieldType field, size_t nbits) {
  const auto ty = makeType<BShrTy>(field, nbits);
  return r.as(ty);
}
}  // namespace

static NdArrayRef wrap_add_bb(SPUContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y) {
  SPU_ENFORCE(x.shape() == y.shape());
  return UnwrapValue(add_bb(ctx, WrapValue(x), WrapValue(y)));
}

NdArrayRef A2B::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto* prg_state = ctx->getState<PrgState>();

  std::vector<NdArrayRef> bshrs;
  const auto bty = makeType<BShrTy>(field);
  for (size_t idx = 0; idx < comm->getWorldSize(); idx++) {
    auto [r0, r1] =
        prg_state->genPrssPair(field, x.shape(), PrgState::GenPrssCtrl::Both);
    auto b = ring_xor(r0, r1).as(bty);

    if (idx == comm->getRank()) {
      ring_xor_(b, x);
    }
    bshrs.push_back(b.as(bty));
  }

  NdArrayRef res = vreduce(bshrs.begin(), bshrs.end(),
                           [&](const NdArrayRef& xx, const NdArrayRef& yy) {
                             return wrap_add_bb(ctx->sctx(), xx, yy);
                           });
  return res.as(bty);
}

NdArrayRef B2A::proc(KernelEvalContext* ctx, const NdArrayRef& x) const {
  size_t n = x.numel();
  size_t num_job = InitOTState(ctx, n);
  size_t work_load = num_job == 0 ? 0 : CeilDiv(n, num_job);

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const auto flatten_x = x.reshape({static_cast<int64_t>(n)});
  NdArrayRef out(x.eltype(), x.shape());

  TiledDispatch(ctx, num_job, [&](int64_t job) {
    auto slice_bgn = std::min<int64_t>(n, job * work_load);
    auto slice_end = std::min<int64_t>(n, slice_bgn + work_load);
    if (slice_bgn == slice_end) {
      return;
    }
    auto out_slice = ctx->getState<BumblebeeOTState>()->get(job)->B2A(
        flatten_x.slice({slice_bgn}, {slice_end}, {1}));
    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.elsize() * out_slice.numel());
  });

  return out.as(makeType<AShrTy>(field));
}

void CommonTypeV::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_TRACE_MPC_DISP(ctx, lhs, rhs);

  const auto* lhs_v = lhs.as<Priv2kTy>();
  const auto* rhs_v = rhs.as<Priv2kTy>();

  ctx->setOutput(makeType<AShrTy>(std::max(lhs_v->field(), rhs_v->field())));
}

NdArrayRef P2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

NdArrayRef A2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

NdArrayRef A2V::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                     size_t rank) const {
  auto* comm = ctx->getState<Communicator>();
  const auto field = in.eltype().as<AShrTy>()->field();
  auto out_ty = makeType<Priv2kTy>(field, rank);

  auto numel = in.numel();

  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    std::vector<ring2k_t> share(numel);
    NdArrayView<ring2k_t> _in(in);
    pforeach(0, numel, [&](int64_t idx) { share[idx] = _in[idx]; });

    std::vector<std::vector<ring2k_t>> shares =
        comm->gather<ring2k_t>(share, rank, "a2v");  // comm => 1, k
    if (comm->getRank() == rank) {
      SPU_ENFORCE(shares.size() == comm->getWorldSize());
      NdArrayRef out(out_ty, in.shape());
      NdArrayView<ring2k_t> _out(out);
      pforeach(0, numel, [&](int64_t idx) {
        ring2k_t s = 0;
        for (auto& share : shares) {
          s += share[idx];
        }
        _out[idx] = s;
      });
      return out;
    } else {
      return makeConstantArrayRef(out_ty, in.shape());
    }
  });
}

NdArrayRef V2A::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto* in_ty = in.eltype().as<Priv2kTy>();
  const size_t owner_rank = in_ty->owner();
  const auto field = in_ty->field();
  auto* prg_state = ctx->getState<PrgState>();
  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_sub(r0, r1).as(makeType<AShrTy>(field));

  if (comm->getRank() == owner_rank) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

void CommonTypeB::evaluate(KernelEvalContext* ctx) const {
  const Type& lhs = ctx->getParam<Type>(0);
  const Type& rhs = ctx->getParam<Type>(1);

  SPU_ENFORCE(lhs == rhs,
              "bumblebee always use same bshare type, lhs={}, rhs={}", lhs,
              rhs);

  ctx->setOutput(lhs);
}

NdArrayRef CastTypeB::proc(KernelEvalContext*, const NdArrayRef& in,
                           const Type& to_type) const {
  SPU_ENFORCE(in.eltype() == to_type,
              "bumblebee always use same bshare type, lhs={}, rhs={}",
              in.eltype(), to_type);
  return in;
}

NdArrayRef B2P::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::XOR, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

NdArrayRef P2B::proc(KernelEvalContext* ctx, const NdArrayRef& in) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  auto* prg_state = ctx->getState<PrgState>();

  auto* comm = ctx->getState<Communicator>();

  auto [r0, r1] =
      prg_state->genPrssPair(field, in.shape(), PrgState::GenPrssCtrl::Both);
  auto x = ring_xor(r0, r1).as(makeType<BShrTy>(field, 0));

  if (comm->getRank() == 0) {
    ring_xor_(x, in);
  }

  return makeBShare(x, field, getNumBits(in));
}
}  // namespace spu::mpc::bumblebee
