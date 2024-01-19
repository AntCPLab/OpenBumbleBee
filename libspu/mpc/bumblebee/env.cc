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

#include "libspu/mpc/bumblebee/env.h"

#include <algorithm>
#include <string>

namespace spu::mpc::bumblebee {
static bool IsEnvOn(const char *name, bool by_default) {
  const char *str = std::getenv(name);
  if (str == nullptr) {
    return by_default;
  }

  std::string s(str);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "1" or s == "on") {
    return true;
  }
  return s == "1" or s == "on";
}

static int GetEnvInt(const char *name) {
  const char *str = std::getenv(name);
  if (str == nullptr) {
    return 0;
  }

  return std::atoi(str);
}

bool TestEnvFlag(EnvFlag e) {
  switch (e) {
    case EnvFlag::SPU_BB_ENABLE_APPROX_LESS_THAN:
      return IsEnvOn("SPU_BB_ENABLE_APPROX_LESS_THAN", true);
    case EnvFlag::SPU_BB_ENABLE_MUL_ERROR:
      return IsEnvOn("SPU_BB_ENABLE_MUL_ERROR", true);
    default:
      return false;
  }
}

int TestEnvInt(EnvFlag e) {
  switch (e) {
    case EnvFlag::SPU_BB_SET_IEQUAL_BITS:
      return GetEnvInt(("SPU_BB_SET_IEQUAL_BITS"));
    case EnvFlag::SPU_BB_SET_OT_TYPE:
      return GetEnvInt(("SPU_BB_SET_OT_TYPE"));
    default:
      break;
  }
  return 0;
}

}  // namespace spu::mpc::bumblebee
