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
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"

namespace spu::mpc::cheetah {

CheetahMul::CheetahMul(std::shared_ptr<yacl::link::Context> lctx,
                       bool allow_high_prob_one_bit_error) {
  impl_ = std::make_shared<CheetahMulImpl>(lctx, allow_high_prob_one_bit_error);
}

CheetahMul::~CheetahMul() = default;

int CheetahMul::Rank() const { return impl_->Rank(); }

size_t CheetahMul::OLEBatchSize() const {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->OLEBatchSize();
}

NdArrayRef CheetahMul::MulOLE(const NdArrayRef &inp, yacl::link::Context *conn,
                              bool is_evaluator, uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->MulOLE(inp, conn, is_evaluator, msg_width_hint);
}

NdArrayRef CheetahMul::MulOLE(const NdArrayRef &inp, bool is_evaluator,
                              uint32_t msg_width_hint) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->MulOLE(inp, nullptr, is_evaluator, msg_width_hint);
}

}  // namespace spu::mpc::cheetah
