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

#pragma once

#include <memory>
#include <mutex>

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"

namespace spu::mpc::cheetah {

using OTUnaryFunc = std::function<NdArrayRef(
    const NdArrayRef& sub, const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTBinaryFunc =
    std::function<NdArrayRef(const NdArrayRef& op0, const NdArrayRef& op1,
                             const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTUnaryFuncWithU8 = std::function<NdArrayRef(
    absl::Span<const uint8_t> op, const std::shared_ptr<BasicOTProtocols>& ot)>;

using OTBinaryFuncWithU8 = std::function<NdArrayRef(
    const NdArrayRef& op0, absl::Span<const uint8_t> op1,
    const std::shared_ptr<BasicOTProtocols>& ot)>;

NdArrayRef DispatchUnaryFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                             OTUnaryFunc func);

NdArrayRef DispatchBinaryFunc(KernelEvalContext* ctx, const NdArrayRef& x,
                              const NdArrayRef& y, OTBinaryFunc func);

NdArrayRef DispatchUnaryFuncWithBatchedInput(KernelEvalContext* ctx,
                                             const NdArrayRef& input,
                                             bool is_batcher,
                                             int64_t batch_size,
                                             OTUnaryFunc func);

NdArrayRef DispatchBinaryFuncWithBatchedInput(KernelEvalContext* ctx,
                                              const NdArrayRef& input,
                                              const NdArrayRef& batched_input,
                                              OTBinaryFunc func);

}  // namespace spu::mpc::cheetah
