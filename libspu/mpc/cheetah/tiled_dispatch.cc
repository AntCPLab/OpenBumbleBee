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

#include "libspu/mpc/cheetah/tiled_dispatch.h"

#include <future>

#include "libspu/mpc/cheetah/state.h"

namespace spu::mpc::cheetah {

namespace {
// Return num_workers for the given size of jobs
size_t InitOTState(KernelEvalContext* ctx, size_t njobs) {
  constexpr size_t kMinWorkSize = 2048;
  if (njobs == 0) {
    return 0;
  }
  auto* comm = ctx->getState<Communicator>();
  auto* ot_state = ctx->getState<CheetahOTState>();
  size_t nworker =
      std::min(ot_state->maximum_instances(), CeilDiv(njobs, kMinWorkSize));
  for (size_t w = 0; w < nworker; ++w) {
    ot_state->LazyInit(comm, w);
  }
  return nworker;
}
}  // namespace

NdArrayRef DispatchUnaryFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                             OTUnaryFunc func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return DispatchUnaryFunc(ctx, x.reshape({numel}), func).reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto slice_input = x.slice({slice_bgn}, {slice_end}, {});
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& input) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input, ot_instance);
        },
        wi, slice_input));
  }

  auto slice_input = x.slice({slice_end}, {numel}, {1});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(slice_input, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef DispatchBinaryFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y, OTBinaryFunc func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  SPU_ENFORCE_EQ(shape, y.shape());
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return DispatchBinaryFunc(ctx, x.reshape({numel}), y.reshape({numel}), func)
        .reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto x_slice = x.slice({slice_bgn}, {slice_end}, {1});
    auto y_slice = y.slice({slice_bgn}, {slice_end}, {1});
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& inp0, const NdArrayRef& inp1) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(inp0, inp1, ot_instance);
        },
        wi, x_slice, y_slice));
  }

  auto x_slice = x.slice({slice_end}, {numel}, {});
  auto y_slice = y.slice({slice_end}, {numel}, {});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, y_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef DispatchUnaryFuncWithBatchedInput(KernelEvalContext* ctx,
                                             const NdArrayRef& x,
                                             bool is_batcher,
                                             int64_t batch_size,
                                             OTUnaryFunc func) {
  Shape shape = x.shape();
  // (lazy) init OT
  int64_t numel = shape.numel();

  if (is_batcher) {
    SPU_ENFORCE(numel % batch_size == 0);
    numel /= batch_size;
  }

  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    Shape oshape = x.shape();
    if (not is_batcher) {
      oshape.insert(oshape.end(), batch_size);
    }
    return DispatchUnaryFuncWithBatchedInput(ctx, x.reshape({x.numel()}),
                                             is_batcher, batch_size, func)
        .reshape(oshape);
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t eidx = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t sidx = wi * workload;
    eidx = std::min(numel, sidx + workload);
    if (is_batcher) {
      sidx *= batch_size;
      eidx *= batch_size;
    }
    auto x_slice = x.slice({sidx}, {eidx}, {});
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& input0) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input0, ot_instance);
        },
        wi, x_slice));
  }
  auto x_slice = x.slice({eidx}, {x.numel()}, {});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  // check size
  int64_t sum = 0;
  for (auto& out_slice : outs) {
    sum += out_slice.numel();
  }
  SPU_ENFORCE_EQ(sum, numel * batch_size);

  Shape oshape = {numel * batch_size};

  NdArrayRef out(outs[0].eltype(), oshape);
  int64_t offset = 0;
  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef DispatchBinaryFuncWithBatchedInput(KernelEvalContext* ctx,
                                              const NdArrayRef& x,
                                              const NdArrayRef& y,
                                              OTBinaryFunc func) {
  Shape shape = x.shape();
  SPU_ENFORCE_EQ(shape.ndim() + 1, y.shape().ndim());
  for (int64_t d = 0; d < shape.ndim(); ++d) {
    SPU_ENFORCE_EQ(shape[d], y.shape()[d]);
  }

  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);
  int64_t batch_size = y.shape().back();
  if (shape.ndim() != 1) {
    return DispatchBinaryFuncWithBatchedInput(
               ctx, x.reshape({numel}), y.reshape({batch_size * numel}), func)
        .reshape(y.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t eidx = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t sidx = wi * workload;
    eidx = std::min(numel, sidx + workload);
    auto x_slice = x.slice({sidx}, {eidx}, {});
    auto y_slice = y.slice({batch_size * sidx}, {batch_size * eidx}, {});

    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& input0, const NdArrayRef& input1) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input0, input1, ot_instance);
        },
        wi, x_slice, y_slice));
  }

  auto x_slice = x.slice({eidx}, {numel}, {});
  auto y_slice = y.slice({batch_size * eidx}, {batch_size * numel}, {});
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, y_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), y.shape());
  int64_t offset = 0;
  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx,
                               absl::Span<const uint8_t> x,
                               OTUnaryFuncWithU8 func) {
  SPU_ENFORCE(not x.empty());
  // (lazy) init OT
  int64_t numel = x.size();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto slice_input = x.subspan(slice_bgn, slice_end - slice_bgn);
    futures.emplace_back(std::async(
        [&](int64_t idx, absl::Span<const uint8_t> input) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(input, ot_instance);
        },
        wi, slice_input));
  }

  auto slice_input = x.subspan(slice_end, numel - slice_end);
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(slice_input, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), {numel});
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

NdArrayRef TiledDispatchOTFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                               absl::Span<const uint8_t> y,
                               OTBinaryFuncWithU8 func) {
  const Shape& shape = x.shape();
  SPU_ENFORCE(shape.numel() > 0);
  SPU_ENFORCE_EQ(shape.numel(), (int64_t)y.size());
  // (lazy) init OT
  int64_t numel = x.numel();
  int64_t nworker = InitOTState(ctx, numel);
  int64_t workload = nworker == 0 ? 0 : CeilDiv(numel, nworker);

  if (shape.ndim() != 1) {
    // TiledDispatchOTFunc over flatten input
    return TiledDispatchOTFunc(ctx, x.reshape({numel}), y, func)
        .reshape(x.shape());
  }

  std::vector<NdArrayRef> outs(nworker);
  std::vector<std::future<void>> futures;

  int64_t slice_end = 0;
  for (int64_t wi = 0; wi + 1 < nworker; ++wi) {
    int64_t slice_bgn = wi * workload;
    slice_end = std::min(numel, slice_bgn + workload);
    auto x_slice = x.slice({slice_bgn}, {slice_end}, {1});
    auto y_slice = y.subspan(slice_bgn, slice_end - slice_bgn);
    futures.emplace_back(std::async(
        [&](int64_t idx, const NdArrayRef& inp0,
            absl::Span<const uint8_t> inp1) {
          auto ot_instance = ctx->getState<CheetahOTState>()->get(idx);
          outs[idx] = func(inp0, inp1, ot_instance);
        },
        wi, x_slice, y_slice));
  }

  auto x_slice = x.slice({slice_end}, {numel}, {});
  auto y_slice = y.subspan(slice_end, numel - slice_end);
  auto ot_instance = ctx->getState<CheetahOTState>()->get(nworker - 1);
  outs[nworker - 1] = func(x_slice, y_slice, ot_instance);

  for (auto&& f : futures) {
    f.get();
  }

  NdArrayRef out(outs[0].eltype(), x.shape());
  int64_t offset = 0;

  for (auto& out_slice : outs) {
    std::memcpy(out.data<std::byte>() + offset, out_slice.data(),
                out_slice.numel() * out.elsize());
    offset += out_slice.numel() * out.elsize();
  }

  return out;
}

}  // namespace spu::mpc::cheetah
