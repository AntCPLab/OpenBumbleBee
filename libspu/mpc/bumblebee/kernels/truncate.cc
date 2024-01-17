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
#include "libspu/mpc/bumblebee/kernels/truncate.h"

#include <future>

#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/bumblebee/arith/common.h"
#include "libspu/mpc/bumblebee/nonlinear/truncate_prot.h"
#include "libspu/mpc/bumblebee/state.h"
#include "libspu/mpc/bumblebee/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::bumblebee {

TruncA::TruncA(OtConfig config) : config_(config) {
  size_t max = BumblebeeOTState::kMaxNumOtInstances;
  SPU_ENFORCE(config.instance_offset < max);
  config_.num_instances =
      std::min(max - config.instance_offset, config.num_instances);
}

NdArrayRef TruncA::proc(KernelEvalContext* ctx, const NdArrayRef& x,
                        size_t bits, SignType sign) const {
  int64_t numel = x.numel();
  NdArrayRef out(x.eltype(), x.shape());
  if (numel == 0) {
    return out;
  }

  int64_t num_job = config_.num_instances == 0 ? InitOTState(ctx, numel)
                                               : config_.num_instances;
  int64_t work_load = CeilDiv(numel, num_job);

  TruncateProtocol::Meta meta;
  meta.signed_arith = true;
  meta.sign = sign;
  meta.shift_bits = bits;
  meta.use_heuristic = true;

  // Operate on 1D array
  auto flatten_x = x.reshape({x.numel()});
  TiledDispatch(ctx, num_job, [&](int64_t job) {
    auto* ot_state = ctx->getState<BumblebeeOTState>();
    auto* comm = ctx->getState<Communicator>();
    ot_state->LazyInit(comm, config_.instance_offset + job);
    auto ot_instance = ot_state->get(config_.instance_offset + job);

    int64_t slice_bgn = std::min(job * work_load, numel);
    int64_t slice_end = std::min(slice_bgn + work_load, numel);
    if (slice_end == slice_bgn) {
      return;
    }

    TruncateProtocol prot(ot_instance);
    auto out_slice =
        prot.Compute(flatten_x.slice({slice_bgn}, {slice_end}, {1}), meta);
    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.numel() * out_slice.elsize());
  });

  return out;
}
}  // namespace spu::mpc::bumblebee