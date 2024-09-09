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

#include "libspu/mpc/cheetah/dialect/cheetor/type.h"

#include <mutex>

#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::cheetor {

void registerTypes() {
  regPV2kTypes();

  static std::once_flag flag;
  std::call_once(flag, []() {
    TypeContext::getTypeContext()->addTypes<AShrTy, PrimeShrTy, BShrTy>();
  });
}

// Switch prime/ashr type
PrimeShrTy::operator AShrTy() { return AShrTy(field_, nbits_); }

AShrTy::operator PrimeShrTy() { return PrimeShrTy(field_, nbits_); }

}  // namespace spu::mpc::cheetor
