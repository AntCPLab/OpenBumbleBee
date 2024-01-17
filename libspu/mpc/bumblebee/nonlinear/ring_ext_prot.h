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

#pragma once

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::bumblebee {

class BasicOTProtocols;

// From [x]_k in 2^k to compute [x]_k' in 2^k' for k' > k
class RingExtProtocol {
 public:
  // For x \in [-2^{k - 2}, 2^{k - 2})
  // 0 <= x + 2^{k - 2} < 2^{k - 1} ie the MSB is always positive
  static constexpr size_t kHeuristicBound = 2;

  struct Meta {
    bool signed_arith = true;
    bool use_heuristic = false;
    SignType sign = SignType::Unknown;

    FieldType src_field;
    FieldType dst_field;
    int64_t src_ring_width;
    int64_t dst_ring_width;
  };

  explicit RingExtProtocol(const std::shared_ptr<BasicOTProtocols> &base);

  ~RingExtProtocol();

  NdArrayRef Compute(const NdArrayRef &inp, Meta meta);

 private:
  NdArrayRef ZeroExtend(const NdArrayRef &inp, const Meta &meta);

  NdArrayRef ComputeWrap(const NdArrayRef &inp, const Meta &meta);

  // w = msbA | msbB
  NdArrayRef MSB0ToWrap(const NdArrayRef &inp, const Meta &meta);

  // w = msbA & msbB
  NdArrayRef MSB1ToWrap(const NdArrayRef &inp, const Meta &meta);

  std::shared_ptr<BasicOTProtocols> basic_ot_prot_ = nullptr;
};

// Input: h0 + h1 = h \mod p
// Output: x0 + x1 = x \mod 2^k
// if `trunc_bits` is set then also to perform 1-bit approximated truncation.
NdArrayRef RingExtPrime(const NdArrayRef &input, const NdArrayRef &prime,
                        FieldType dst_field, std::optional<size_t> trunc_bits,
                        const std::shared_ptr<BasicOTProtocols> &base_ot);

}  // namespace spu::mpc::bumblebee
