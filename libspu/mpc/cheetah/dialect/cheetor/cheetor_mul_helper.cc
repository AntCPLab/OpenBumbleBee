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

#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_helper.h"

#include <cstddef>
#include <future>
#include <memory>

#include "seal/decryptor.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

namespace spu::mpc::cheetor {

CheetorMulHelper::CheetorMulHelper(
    std::shared_ptr<cheetah::SIMDMulProt>& simd_mul_prot,
    const seal::SEALContext& context)
    : simd_mul_prot_(simd_mul_prot), context_(context) {
  SPU_ENFORCE(simd_mul_prot_ != nullptr);
  SPU_ENFORCE(context.parameters_set());
}

void CheetorMulHelper::MulPrimeShareSend(
    absl::Span<const uint64_t> hshr, const seal::SecretKey& sym_enc_key,
    const std::shared_ptr<yacl::link::Context>& conn,
    absl::Span<uint64_t> out) const {
  SPU_ENFORCE_EQ(hshr.size(), out.size());

  size_t simd_lane = simd_mul_prot_->SIMDLane();
  size_t num_splits = cheetah::CeilDiv(hshr.size(), simd_lane);

  std::vector<seal::Ciphertext> ct(num_splits);
  auto out_ct = absl::MakeSpan(ct);

  yacl::parallel_for(0, num_splits, [&](int64_t job_bgn, int64_t job_end) {
    std::vector<seal::Plaintext> pt(job_end - job_bgn);

    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t split_id = job_id % num_splits;
      int64_t slice_bgn = split_id * simd_lane;
      int64_t slice_n = std::min<int64_t>(simd_lane, hshr.size() - slice_bgn);

      simd_mul_prot_->EncodeSingle(hshr.subspan(slice_bgn, slice_n),
                                   pt[job_id - job_bgn]);
    }

    simd_mul_prot_->SymEncrypt(pt, sym_enc_key, context_, true,
                               out_ct.subspan(job_bgn, job_end - job_bgn));
  });

  // dont send too fast
  constexpr size_t kCtAsyncParallel = 16;
  int nxt_rank = conn->NextRank();
  for (size_t i = 0; i < num_splits; i += kCtAsyncParallel) {
    int64_t this_batch = std::min(num_splits - i, kCtAsyncParallel);

    for (int64_t j = 0; j < this_batch; ++j) {
      if (j + 1 < this_batch) {
        conn->SendAsync(nxt_rank, cheetah::EncodeSEALObject(ct[i + j]),
                        fmt::format("send ct[{}] to rank{}", i + j, nxt_rank));
      } else {
        conn->Send(nxt_rank, cheetah::EncodeSEALObject(ct[i + j]),
                   fmt::format("send ct[{}] to rank{}", i + j, nxt_rank));
      }
    }
  }

  // recv result
  for (size_t i = 0; i < num_splits; ++i) {
    auto recv = conn->Recv(nxt_rank,
                           fmt::format("recv ct[{}] from rank{}", i, nxt_rank));
    cheetah::DecodeSEALObject(recv, context_, &ct[i]);
  }

  // decrypt and decode
  seal::Decryptor decryptor(context_, sym_enc_key);
  yacl::parallel_for(0, num_splits, [&](int64_t job_bgn, int64_t job_end) {
    seal::Plaintext pt;
    for (int64_t job_id = job_bgn; job_id < job_end; ++job_id) {
      int64_t split_id = job_id % num_splits;
      int64_t slice_bgn = split_id * simd_lane;
      int64_t slice_n = std::min<int64_t>(simd_lane, hshr.size() - slice_bgn);
      decryptor.decrypt(ct[job_id], pt);

      simd_mul_prot_->DecodeSingle(pt, out.subspan(slice_bgn, slice_n));
    }

    seal::util::seal_memzero(pt.data(), sizeof(uint64_t) * pt.coeff_count());
  });
}

void CheetorMulHelper::MulPrimeShareRecv(
    absl::Span<const uint64_t> hshr, const seal::PublicKey& peer_pub_key,
    const std::shared_ptr<yacl::link::Context>& conn,
    absl::Span<uint64_t> out) const {
  SPU_ENFORCE_EQ(hshr.size(), out.size());

  const size_t simd_lane = simd_mul_prot_->SIMDLane();
  const size_t num_splits = cheetah::CeilDiv(hshr.size(), simd_lane);
  const int nxt_rank = conn->NextRank();

  // recv ct
  std::vector<seal::Ciphertext> recv_ct(num_splits);
  auto io_task = std::async(std::launch::async, [&]() {
    for (size_t i = 0; i < num_splits; ++i) {
      auto recv = conn->Recv(
          nxt_rank, fmt::format("recv ct[{}] from rank{}", i, nxt_rank));
      cheetah::DecodeSEALObject(recv, context_, &recv_ct[i]);
    }
  });

  // encode pt
  std::vector<seal::Plaintext> _pt(num_splits);
  auto pt = absl::MakeSpan(_pt);
  yacl::parallel_for(0, num_splits, [&](int64_t job_bgn, int64_t job_end) {
    int64_t njob = job_end - job_bgn;
    int64_t slice_bgn = job_bgn * simd_lane;
    int64_t slice_end =
        std::min<int64_t>(slice_bgn + njob * simd_lane, hshr.size());

    simd_mul_prot_->EncodeBatch(hshr.subspan(slice_bgn, slice_end - slice_bgn),
                                pt.subspan(job_bgn, njob));
  });

  // sample ranom mask as output
  cheetah::EnableCPRNG cprng;
  cprng.UniformPrime(simd_mul_prot_->modulus(), out);

  // ct-pt mul
  io_task.get();

  auto ct = absl::MakeSpan(recv_ct);
  yacl::parallel_for(0, num_splits, [&](int64_t job_bgn, int64_t job_end) {
    int64_t njob = job_end - job_bgn;
    int64_t slice_bgn = job_bgn * simd_lane;
    int64_t slice_end =
        std::min<int64_t>(slice_bgn + njob * simd_lane, hshr.size());

    simd_mul_prot_->MulThenReshareInplace(
        ct.subspan(job_bgn, njob), pt.subspan(job_bgn, njob),
        out.subspan(slice_bgn, slice_end - slice_bgn), peer_pub_key, context_);
  });

  // dont send too fast
  constexpr size_t kCtAsyncParallel = 16;
  for (size_t i = 0; i < num_splits; i += kCtAsyncParallel) {
    int64_t this_batch = std::min(num_splits - i, kCtAsyncParallel);

    for (int64_t j = 0; j < this_batch; ++j) {
      if (j + 1 < this_batch) {
        conn->SendAsync(nxt_rank, cheetah::EncodeSEALObject(ct[i + j]),
                        fmt::format("send ct[{}] to rank{}", i + j, nxt_rank));
      } else {
        conn->Send(nxt_rank, cheetah::EncodeSEALObject(ct[i + j]),
                   fmt::format("send ct[{}] to rank{}", i + j, nxt_rank));
      }
    }
  }
}

}  // namespace spu::mpc::cheetor
