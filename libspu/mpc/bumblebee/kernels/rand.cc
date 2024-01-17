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

#include "libspu/mpc/bumblebee/kernels/rand.h"

#include "libspu/mpc/bumblebee/state.h"
#include "libspu/mpc/bumblebee/type.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::bumblebee {

NdArrayRef RandA::proc(KernelEvalContext* ctx, const Shape& shape) const {
  auto* prg_state = ctx->getState<PrgState>();
  const auto field = ctx->getState<Z2kState>()->getDefaultField();
  return ring_rshift(prg_state->genPriv(field, shape), 2)
      .as(makeType<AShrTy>(field));
}

}  // namespace spu::mpc::bumblebee
