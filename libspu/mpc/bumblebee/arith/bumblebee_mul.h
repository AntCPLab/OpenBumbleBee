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
#include "libspu/mpc/bumblebee/rlwe/types.h"

namespace spu::mpc::bumblebee {

class BumblebeeMul;

// Note(lwj): export some APIs for other use
class BumblebeeMulImpl {
 public:
  explicit BumblebeeMulImpl(std::shared_ptr<yacl::link::Context> lctx,
                            bool allow_high_prob_one_bit_error = false);

  ~BumblebeeMulImpl();

  void Initialize(FieldType field, uint32_t msg_width = 0);

  int Rank() const;

  size_t OLEBatchSize() const;

  std::vector<RLWECt> EncryptArray(const NdArrayRef& inp);

  NdArrayRef MultiplyThenMask(const NdArrayRef& inp,
                              absl::Span<RLWECt> recv_ct);

  std::vector<RLWECt> RecvEncryptedArray(FieldType field, int64_t numel,
                                         yacl::link::Context* conn = nullptr);

  NdArrayRef RecvMulArray(FieldType field, int64_t numel,
                          yacl::link::Context* conn = nullptr);

 private:
  friend class BumblebeeMul;
  NdArrayRef MulOLE(const NdArrayRef& inp, yacl::link::Context* conn,
                    bool is_evaluator, uint32_t msg_width_hint = 0);

  struct Impl;
  std::unique_ptr<Impl> impl_{nullptr};
};

// Implementation for 1-bit approximated Mul
// Ref: Lu et al. "BumbleBee: Secure Two-party Inference Framework for Large
// Transformers"
//  https://eprint.iacr.org/2023/1678
class BumblebeeMul {
 public:
  explicit BumblebeeMul(std::shared_ptr<yacl::link::Context> lctx,
                        bool allow_high_prob_one_bit_error = false);

  ~BumblebeeMul();

  BumblebeeMul& operator=(const BumblebeeMul&) = delete;

  BumblebeeMul(const BumblebeeMul&) = delete;

  BumblebeeMul(BumblebeeMul&&) = delete;

  void LazyInitKeys(FieldType field, uint32_t msg_width_hint = 0);

  NdArrayRef MulOLE(const NdArrayRef& inp, yacl::link::Context* conn,
                    bool is_evaluator, uint32_t msg_width_hint = 0);

  NdArrayRef MulOLE(const NdArrayRef& inp, bool is_evaluator,
                    uint32_t msg_width_hint = 0);

  BumblebeeMulImpl* getImpl() { return impl_.get(); }

  int Rank() const;

  size_t OLEBatchSize() const;

 private:
  std::shared_ptr<BumblebeeMulImpl> impl_{nullptr};
};

}  // namespace spu::mpc::bumblebee
