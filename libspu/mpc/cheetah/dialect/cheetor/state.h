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

#include "libspu/core/object.h"
#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"

namespace spu::mpc::cheetor {

class CheetorMulState : public State {
 private:
  std::shared_ptr<CheetorMulProt> mul_prot_;

 public:
  static constexpr char kBindName[] = "CheetorMul";

  explicit CheetorMulState(const std::shared_ptr<yacl::link::Context>& conn) {
    mul_prot_ = std::make_shared<CheetorMulProt>(conn);
  }

  ~CheetorMulState() override = default;

  std::shared_ptr<CheetorMulProt> GetMulProt() { return mul_prot_; }
};

}  // namespace spu::mpc::cheetor
