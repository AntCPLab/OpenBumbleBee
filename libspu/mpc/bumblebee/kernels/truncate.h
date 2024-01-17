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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::bumblebee {

class TruncA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  struct OtConfig {
    size_t instance_offset;
    size_t num_instances;
  };

  explicit TruncA(OtConfig config = {0UL, 0UL});

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }

 private:
  OtConfig config_;
};
}  // namespace spu::mpc::bumblebee