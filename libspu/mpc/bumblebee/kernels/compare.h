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

#include "libspu/mpc/kernel.h"

namespace spu::mpc::bumblebee {

class MsbA2B : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_a2b";

  MsbA2B(size_t nbits = 0) : nbits_(nbits) {}

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x) const override;

 private:
  size_t nbits_;
};

class EqualAA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_aa";

  EqualAA(size_t nbits = 0) : nbits_(nbits) {}

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;

 private:
  size_t nbits_;
};

class EqualAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "equal_ap";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LessAP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "f_less_ap";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class LessPA : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "f_less_pa";

  Kind kind() const override { return Kind::Dynamic; }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

}  // namespace spu::mpc::bumblebee