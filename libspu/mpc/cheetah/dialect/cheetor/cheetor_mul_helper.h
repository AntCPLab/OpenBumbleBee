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

#include "yacl/link/context.h"

#include "libspu/mpc/cheetah/arith/simd_mul_prot.h"

namespace spu::mpc::cheetor {

class CheetorMulHelper {
 public:
  explicit CheetorMulHelper(
      std::shared_ptr<cheetah::SIMDMulProt>& simd_mul_prot,
      const seal::SEALContext& context);

  ~CheetorMulHelper() = default;

  // x, y in [0, p)
  // compute x * y mod p
  void MulPrimeShareSend(absl::Span<const uint64_t> hshr,
                         const seal::SecretKey& sym_enc_key,
                         const std::shared_ptr<yacl::link::Context>& conn,
                         absl::Span<uint64_t> out) const;
  // x, y in [0, p)
  // compute x * y mod p
  void MulPrimeShareRecv(absl::Span<const uint64_t> hshr,
                         const seal::PublicKey& peer_pub_key,
                         const std::shared_ptr<yacl::link::Context>& conn,
                         absl::Span<uint64_t> out) const;

 private:
  // temp hold, no copy
  std::shared_ptr<cheetah::SIMDMulProt>& simd_mul_prot_;
  const seal::SEALContext& context_;
};

}  // namespace spu::mpc::cheetor
