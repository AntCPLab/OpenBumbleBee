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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::cheetor {

class SquareA : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "square_a";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;
};

class MulAA : public BinaryKernel {
 private:
  NdArrayRef mulDirectly(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const;

 public:
  static constexpr char kBindName[] = "mul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class Power4 : public Kernel {
 public:
  static constexpr char kBindName[] = "power4_a";

  Kind kind() const override { return Kind::Dynamic; }

  // Given x in [0, 2^k)
  // Compute x^2, x^3, x^4 mod 2^k
  // The multiplications are done in the specified prime field.
  // The final results are converted to modulus 2^k.
  std::array<NdArrayRef, 3> proc(KernelEvalContext* ctx, const NdArrayRef& x,
                                 FieldType target_field, bool is_fxp) const;
};

// Given x mod p, compute x/2^d and results at mod 2^k
class TruncPrimeToRing : public Kernel {
 public:
  static constexpr char kBindName[] = "trunc_p2r";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  size_t trunc_bits, FieldType target_ring,
                  size_t target_ring_width) const;

  TruncLsbRounding lsbRounding() const {
    return TruncLsbRounding::Probabilistic;
  }
};

// 1-bit approximated truncation with positive heuristic
class TruncPrA : public TruncAKernel {
 public:
  static constexpr char kBindName[] = "trunc_a";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x, size_t bits,
                  SignType sign) const override;

  bool hasMsbError() const override { return false; }

  TruncLsbRounding lsbRounding() const override {
    return TruncLsbRounding::Probabilistic;
  }
};

}  // namespace spu::mpc::cheetor
