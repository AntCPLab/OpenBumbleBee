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

class MatMulAP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ap";

  ce::CExpr latency() const override { return ce::Const(0); }

  ce::CExpr comm() const override { return ce::Const(0); }

  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulVVS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_vvs";

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulAV : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_av";

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class MatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_aa";

  Kind kind() const override { return Kind::Dynamic; }
  // LHS: m x k
  // RHS: k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class BatchMatMulAA : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "batch_mmul_aa";

  Kind kind() const override { return Kind::Dynamic; }

  // override the shape checking
  void evaluate(KernelEvalContext* ctx) const override;
  // LHS: b x m x k
  // RHS: b x k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

class BatchMatMulAV : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "batch_mmul_av";

  Kind kind() const override { return Kind::Dynamic; }

  // override the shape checking
  void evaluate(KernelEvalContext* ctx) const override;
  // LHS: b x m x k
  // RHS: b x k x n
  NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& y) const override;
};

}  // namespace spu::mpc::bumblebee