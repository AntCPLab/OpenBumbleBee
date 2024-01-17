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
#include <mutex>

#include "libspu/core/object.h"
#include "libspu/mpc/bumblebee/arith/bumblebee_dot.h"
#include "libspu/mpc/bumblebee/arith/bumblebee_mul.h"
#include "libspu/mpc/bumblebee/ot/basic_ot_prot.h"

namespace spu::mpc::bumblebee {

// Return num_workers for the given size of jobs
size_t InitOTState(KernelEvalContext* ctx, size_t njobs);

// Call func(idx) for idx = 0, 1, ..., n - 1
void TiledDispatch(KernelEvalContext* ctx, int64_t njobs,
                   const std::function<void(int64_t)>& func);

class BumblebeeMulState : public State {
 private:
  mutable std::mutex lock_;
  // a[2] = a[0] * a[1]
  mutable int64_t cached_sze_{0};
  FieldType field_{FT_INVALID};
  NdArrayRef cached_beaver_[3];
  NdArrayRef cached_square_[2];

  std::unordered_map<FieldType, std::array<NdArrayRef, 3>> beaver_cache_;
  std::unordered_map<FieldType, std::array<NdArrayRef, 2>> square_cache_;

  std::unique_ptr<BumblebeeMul> mul_prot_;
  std::shared_ptr<yacl::link::Context> duplx_;

  // NOTE(juhou): make sure the lock is obtained
  void makeSureCacheSize(FieldType, int64_t numel);

  explicit BumblebeeMulState(std::unique_ptr<BumblebeeMul> mul_prot)
      : mul_prot_(std::move(mul_prot)) {}

 public:
  static constexpr char kBindName[] = "BumblebeeMul";

  explicit BumblebeeMulState(const std::shared_ptr<yacl::link::Context>& lctx,
                             bool allow_mul_error = false) {
    mul_prot_ = std::make_unique<BumblebeeMul>(lctx, allow_mul_error);
    duplx_ = lctx->Spawn();
  }

  ~BumblebeeMulState() override = default;

  BumblebeeMul* get() { return mul_prot_.get(); }

  std::shared_ptr<yacl::link::Context> duplx() { return duplx_; }

  std::array<NdArrayRef, 3> TakeCachedBeaver(FieldType field, int64_t num);

  std::array<NdArrayRef, 2> TakeCachedSquare(FieldType field, int64_t num);
};

class BumblebeeDotState : public State {
 private:
  std::unique_ptr<BumblebeeDot> dot_prot_;

  explicit BumblebeeDotState(std::unique_ptr<BumblebeeDot> dot_prot)
      : dot_prot_(std::move(dot_prot)) {}

 public:
  static constexpr char kBindName[] = "BumblebeeDot";

  explicit BumblebeeDotState(const std::shared_ptr<yacl::link::Context>& lctx,
                             bool enable_matmul_pack = true) {
    dot_prot_ = std::make_unique<BumblebeeDot>(lctx, enable_matmul_pack);
  }

  ~BumblebeeDotState() override = default;

  BumblebeeDot* get() { return dot_prot_.get(); }
};

class BumblebeeOTState : public State {
 private:
  using ProtPtr = std::shared_ptr<BasicOTProtocols>;

  mutable std::mutex lock_;
  size_t instance_allocated_ = 0;
  std::vector<ProtPtr> basic_ot_prot_;

 public:
  static constexpr char kBindName[] = "BumblebeeOT";
  static constexpr size_t kMaxNumOtInstances = 24;

  explicit BumblebeeOTState() : basic_ot_prot_(kMaxNumOtInstances) {}

  ~BumblebeeOTState() override {
    SPDLOG_INFO("BumbleBee allocated {} OT instances", instance_allocated_);
  }

  void LazyInit(Communicator* comm, size_t idx = 0) {
    SPU_ENFORCE(idx < kMaxNumOtInstances, "idx={} out-of-bound", idx);
    std::lock_guard guard(lock_);
    if (basic_ot_prot_[idx]) {
      return;
    }
    // NOTE(lwj): create a separated link for OT
    // We **do not** block on the OT link since the message volume is small for
    // LPN-based OTe
    auto link = comm->lctx()->Spawn();
    link->SetThrottleWindowSize(0);
    auto _comm = std::make_shared<Communicator>(std::move(link));
    basic_ot_prot_[idx] = std::make_shared<BasicOTProtocols>(std::move(_comm));
    instance_allocated_ += 1;
  }

  std::shared_ptr<BasicOTProtocols> get(size_t idx = 0) {
    SPU_ENFORCE(idx < kMaxNumOtInstances, "idx={} out-of-bound", idx);
    SPU_ENFORCE(basic_ot_prot_[idx], "call LazyInit first");
    return basic_ot_prot_[idx];
  }
};

}  // namespace spu::mpc::bumblebee
