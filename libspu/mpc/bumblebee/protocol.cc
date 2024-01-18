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

#include "libspu/mpc/bumblebee/protocol.h"
// FIXME: both emp-tools & openssl defines AES_KEY, hack the include order to
// avoid compiler error.
#include "libspu/mpc/common/prg_state.h"
//

#include "libspu/mpc/bumblebee/env.h"
#include "libspu/mpc/bumblebee/kernels/kernels.h"
#include "libspu/mpc/bumblebee/state.h"
#include "libspu/mpc/bumblebee/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/standard_shape/protocol.h"

namespace spu::mpc {

void regBumblebeeProtocol(SPUContext* ctx,
                          const std::shared_ptr<yacl::link::Context>& lctx) {
  bumblebee::registerTypes();

  // add communicator
  ctx->prot()->addState<Communicator>(lctx);

  // register random states & kernels.
  ctx->prot()->addState<PrgState>(lctx);

  // add Z2k state.
  ctx->prot()->addState<Z2kState>(ctx->config().field());

  // add Bumblebee states
  bool allow_mul_err =
      bumblebee::TestEnvFlag(bumblebee::EnvFlag::SPU_BB_ENABLE_MUL_ERROR);
  bool allow_matmul_pack =
      bumblebee::TestEnvFlag(bumblebee::EnvFlag::SPU_BB_ENABLE_MMUL_PACK);

  ctx->prot()->addState<bumblebee::BumblebeeMulState>(lctx, allow_mul_err);
  ctx->prot()->addState<bumblebee::BumblebeeDotState>(lctx, allow_matmul_pack);
  ctx->prot()->addState<bumblebee::BumblebeeOTState>();

  // register public kernels.
  regPV2kKernels(ctx->prot());

  // Register standard shape ops
  regStandardShapeOps(ctx);

  // register arithmetic & binary kernels
  ctx->prot()
      ->regKernel<
          bumblebee::P2A, bumblebee::A2P, bumblebee::V2A, bumblebee::A2V,  //
          bumblebee::B2P, bumblebee::P2B, bumblebee::A2B, bumblebee::B2A,  //
          bumblebee::NotA,                                                 //
          bumblebee::AddAP, bumblebee::AddAA,                              //
          bumblebee::MulAP, bumblebee::MulAA, bumblebee::MulA1B,           //
          bumblebee::EqualAA, bumblebee::EqualAP,                          //
          bumblebee::MatMulAP, bumblebee::MatMulAA, bumblebee::MatMulAV,   //
          bumblebee::BatchMatMulAA, bumblebee::BatchMatMulAV,              //
          bumblebee::LShiftA, bumblebee::ARShiftB, bumblebee::LShiftB,
          bumblebee::RShiftB,                                        //
          bumblebee::BitrevB,                                        //
          bumblebee::TruncA,                                         //
          bumblebee::MsbA2B,                                         //
          bumblebee::CommonTypeB, bumblebee::CommonTypeV,            //
          bumblebee::CastTypeB, bumblebee::AndBP, bumblebee::AndBB,  //
          bumblebee::XorBP, bumblebee::XorBB,                        //
          bumblebee::RandA>();
}

std::unique_ptr<SPUContext> makeBumblebeeProtocol(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  bumblebee::registerTypes();

  auto ctx = std::make_unique<SPUContext>(conf, lctx);

  regBumblebeeProtocol(ctx.get(), lctx);

  return ctx;
}

}  // namespace spu::mpc
