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

#include "seal/context.h"
#include "yacl/link/link.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"

namespace seal {
class SecretKey;
class PublicKey;
}  // namespace seal

namespace spu::mpc::cheetah {
class SIMDMulProt;
}

namespace spu::mpc::cheetor {

class CheetorMulProt {
 private:
  using SIMDMulProtPtr = std::shared_ptr<spu::mpc::cheetah::SIMDMulProt>;
  // prime -> instance mapping
  std::unordered_map<uint64_t, SIMDMulProtPtr> simd_mul_instances_;

  // prime -> keys mapping
  std::unordered_map<uint64_t, seal::SEALContext> contexts_;
  std::unordered_map<uint64_t, std::shared_ptr<seal::SecretKey>> secret_keys_;
  std::unordered_map<uint64_t, std::shared_ptr<seal::PublicKey>> peer_pub_keys_;

  std::shared_ptr<yacl::link::Context> conn_ = nullptr;
  std::shared_ptr<yacl::link::Context> duplx_ = nullptr;

 public:
  explicit CheetorMulProt(const std::shared_ptr<yacl::link::Context>& conn);

  void LazyInit(FieldType ft);

  static std::vector<uint64_t> GetWorkingPrimes(FieldType ft);

  // the product of the working primes
  static NdArrayRef GetPrimesProduct(FieldType ft);

  static size_t PrimeNumBits(FieldType ft);

  // Given x \in [0, 2^k)
  // compute x^2 mod p
  NdArrayRef RingShareToPrimeShareSquare(const NdArrayRef& x_ring_share);

  // Given x, y \in [0, 2^k)
  // compute x * y mod p
  NdArrayRef RingShareToPrimeShareMul(const NdArrayRef& x_ring_share,
                                      const NdArrayRef& y_ring_share);

  // Given x \in [0, p)
  // compute x^2 mod p
  NdArrayRef PrimeShareSquare(absl::Span<const uint64_t> x_prime_share,
                              FieldType ft);

  // Given x, y in \[0, p) compuote <x * y> mod p
  NdArrayRef MulToPrime(absl::Span<const uint64_t> x, FieldType ft);

  // Given x, y \in [0, p)
  // compute x * y mod p
  NdArrayRef PrimeShareMul(absl::Span<const uint64_t> x_prime_share,
                           absl::Span<const uint64_t> y_prime_share,
                           FieldType ft);

  ~CheetorMulProt();
};

}  // namespace spu::mpc::cheetor
