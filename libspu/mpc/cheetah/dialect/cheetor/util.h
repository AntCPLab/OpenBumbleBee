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

#include "seal/modulus.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::cheetor {

// Given x0 + x1 = x mod 2^k
// Compute h0 + h1 = x mod p with probability > 1 - |x|/2^k
void ProbConvRing2kShareToPrimeShare(const NdArrayRef& inp_share,
                                     absl::Span<uint64_t> oup_share,
                                     const seal::Modulus& prime, int rank);
}  // namespace spu::mpc::cheetor
