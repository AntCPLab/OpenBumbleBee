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

#include "libspu/mpc/cheetah/kernels/bits.h"

#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
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

NdArrayRef AndBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE_EQ(lhs.shape(), rhs.shape());

  int64_t numel = lhs.numel();
  NdArrayRef out(lhs.eltype(), lhs.shape());
  if (numel == 0) {
    return out;
  }

  int64_t nworker = InitOTState(ctx, numel);
  int64_t work_load = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  auto flat_lhs = lhs.reshape({lhs.numel()});
  auto flat_rhs = rhs.reshape({rhs.numel()});
  yacl::parallel_for(0, nworker, 1, [&](int64_t bgn, int64_t end) {
    for (int64_t job = bgn; job < end; ++job) {
      int64_t slice_bgn = std::min(numel, job * work_load);
      int64_t slice_end = std::min(numel, slice_bgn + work_load);
      if (slice_bgn == slice_end) {
        break;
      }

      auto out_slice = ctx->getState<CheetahOTState>()->get(job)->BitwiseAnd(
          flat_lhs.slice({slice_bgn}, {slice_end}, {1}),
          flat_rhs.slice({slice_bgn}, {slice_end}, {1}));
      std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                  out_slice.elsize() * out_slice.numel());
    }
  });

  return out;
}

NdArrayRef AndBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.shape() == rhs.shape());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::min(getNumBits(lhs), getNumBits(rhs));
  NdArrayRef out(makeType<BShrTy>(field, out_nbits), lhs.shape());

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _lhs(lhs);
    NdArrayView<ring2k_t> _rhs(rhs);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, lhs.numel(),
             [&](int64_t idx) { _out[idx] = _lhs[idx] & _rhs[idx]; });
  });
  return out;
}

NdArrayRef XorBP::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  auto* comm = ctx->getState<Communicator>();

  const auto field = lhs.eltype().as<Ring2k>()->field();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));

  if (comm->getRank() == 0) {
    return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
  }

  return makeBShare(lhs, field, out_nbits);
}

NdArrayRef XorBB::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                       const NdArrayRef& rhs) const {
  SPU_ENFORCE(lhs.numel() == rhs.numel());

  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  const size_t out_nbits = std::max(getNumBits(lhs), getNumBits(rhs));
  return makeBShare(ring_xor(lhs, rhs), field, out_nbits);
}

NdArrayRef BitrevB::proc(KernelEvalContext*, const NdArrayRef& in, size_t start,
                         size_t end) const {
  const auto field = in.eltype().as<Ring2k>()->field();

  SPU_ENFORCE(start <= end);
  SPU_ENFORCE(end <= SizeOf(field) * 8);
  const size_t out_nbits = std::max(getNumBits(in), end);

  // TODO: more accurate bits.
  return makeBShare(ring_bitrev(in, start, end), field, out_nbits);
}

NdArrayRef BitIntlB::proc(KernelEvalContext*, const NdArrayRef& in,
                          size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = BitIntl<ring2k_t>(_in[idx], stride, nbits);
    });
  });

  return out;
}

NdArrayRef BitDeintlB::proc(KernelEvalContext*, const NdArrayRef& in,
                            size_t stride) const {
  const auto field = in.eltype().as<Ring2k>()->field();
  const auto nbits = getNumBits(in);
  SPU_ENFORCE(absl::has_single_bit(nbits));

  NdArrayRef out(in.eltype(), in.shape());
  auto numel = in.numel();

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    NdArrayView<ring2k_t> _in(in);
    NdArrayView<ring2k_t> _out(out);

    pforeach(0, numel, [&](int64_t idx) {
      _out[idx] = BitDeintl<ring2k_t>(_in[idx], stride, nbits);
    });
  });

  return out;
}
}  // namespace spu::mpc::cheetah