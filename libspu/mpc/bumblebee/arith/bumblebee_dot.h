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

#include "yacl/link/context.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/bumblebee/arith/common.h"

namespace spu::mpc::bumblebee {

// Implementation for MatMul with ciphertext packing
// Ref: Lu et al. "BumbleBee: Secure Two-party Inference Framework for Large
// Transformers"
//  https://eprint.iacr.org/2023/1678
class BumblebeeDot {
 public:
  enum class Role { encryptor, evaluator, dynamic };

  explicit BumblebeeDot(const std::shared_ptr<yacl::link::Context>& lctx,
                        bool enable_matmul_pack = true);

  ~BumblebeeDot();

  BumblebeeDot& operator=(const BumblebeeDot&) = delete;

  BumblebeeDot(const BumblebeeDot&) = delete;

  BumblebeeDot(BumblebeeDot&&) = delete;

  NdArrayRef DotOLE(const NdArrayRef& inp, const Shape3D& dim3,
                    bool is_self_lhs, Role role = Role::dynamic);

  // LHS.shape MxK, RHS.shape KxL => MxL
  NdArrayRef DotOLE(const NdArrayRef& inp, yacl::link::Context* conn,
                    const Shape3D& dim3, bool is_self_lhs,
                    Role role = Role::dynamic);

  // LHS.shape BxMxK, RHS.shape BxKxL => BxMxL
  NdArrayRef BatchDotOLE(const NdArrayRef& inp, yacl::link::Context* conn,
                         const Shape4D& dim4, bool is_self_lhs);

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_{nullptr};
};

}  // namespace spu::mpc::bumblebee
