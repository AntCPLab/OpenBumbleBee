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

#include "spu/psi/psi.h"

namespace spu::psi {

std::shared_ptr<PsiExecutorBase> BuildPsiExecutor(const std::any& opts) {
  std::shared_ptr<PsiExecutorBase> executor;
  if (opts.type() == typeid(LegacyPsiOptions)) {
    auto op = std::any_cast<LegacyPsiOptions>(opts);
    executor.reset(new LegacyPsiExecutor(op));
  } else {
    YASL_THROW("unknow opts type {}", opts.type().name());
  }

  return executor;
}

}  // namespace spu::psi