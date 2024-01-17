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

namespace spu::mpc::bumblebee {

enum class EnvFlag {
  // enable approximately less than between a share and a plaintext
  SPU_BB_ENABLE_APPROX_LESS_THAN,

  // enable 1bit error in Mul
  SPU_BB_ENABLE_MUL_ERROR,

  // enable ciphertext packing
  SPU_BB_ENABLE_MMUL_PACK,

  // use emp/ferret instead of yacl/ferret
  SPU_BB_ENABLE_EMP_FERRET,

  // set the max bits for i_equal
  SPU_BB_SET_IEQUAL_BITS,
};

bool TestEnvFlag(EnvFlag e);

// If not exist return 0
int TestEnvInt(EnvFlag e);

}  // namespace spu::mpc::bumblebee