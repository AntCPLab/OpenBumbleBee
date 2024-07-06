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

#include <array>

#include "libspu/core/context.h"

namespace spu::mpc::cheetah {

// Compute a batch of less than between a secret and a batch of public value
// [x], y0, y1, ..., yB => less(x < y0), less(x < y1), .., less(x < yB)
std::vector<NdArrayRef> BatchLessThan(KernelEvalContext* kctx,
                                      const NdArrayRef& x,
                                      absl::Span<const float> y);
}  // namespace spu::mpc::cheetah
