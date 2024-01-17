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
namespace spu::mpc {
class Communicator;

namespace bumblebee {

class BumblebeeMul;
class TruncateProtocol;

class PowersCalculator {
 public:
  PowersCalculator();

  // [x] => [x^2], [x^4] and [x^3]
  std::array<NdArrayRef, 3> ComputeUpToPower4(const NdArrayRef& x);

 private:
  std::shared_ptr<Communicator> comm_;
  std::shared_ptr<BumblebeeMul> mul_prot_;
  std::shared_ptr<TruncateProtocol> trunc_prot_;
};
}  // namespace bumblebee

}  // namespace spu::mpc