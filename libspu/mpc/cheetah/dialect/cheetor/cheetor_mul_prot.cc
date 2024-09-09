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

#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"

#include <future>
#include <memory>

#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "yacl/base/buffer.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/trace.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/simd_mul_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_helper.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/util.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetor {

namespace {
template <typename T>
T GetPlainPrime(int idx);

template <>
uint32_t GetPlainPrime(int) {
  // 31bits
  return 2147352577ULL;
}

template <>
uint64_t GetPlainPrime(int idx) {
  return 1152921504606683137ULL;
}

template <>
uint128_t GetPlainPrime(int idx) {
  if (idx == 0) {
    // 59bits
    return 1152921504606584833ULL;
  }
  // 59bits
  return 1152921504606683137ULL;
}

size_t NumCRT(FieldType ft) {
  switch (ft) {
    default:
    case FM32:
    case FM64:
      return 1;
    case FM128:
      return 2;
  }
}

constexpr size_t SIMDLane(FieldType ft) { return 8192UL; }

seal::EncryptionParameters DecideSEALParameters(FieldType ft) {
  size_t poly_deg = SIMDLane(ft);
  auto scheme_type = seal::scheme_type::bfv;
  auto parms = seal::EncryptionParameters(scheme_type);

  std::vector<int> modulus_bits;

  if (ft == FM32) {
    // one 31-bit modulus for FM32
    modulus_bits = {45, 33, 45};
  } else if (ft == FM128) {
    // two 60-bit modulus for FM64
    modulus_bits = {55, 59, 59};
  } else {
    if (NumCRT(FM64) == 1) {
      // one 60-bit modulus for FM64
      modulus_bits = {55, 59, 59};
    } else {
      // two 31-bit modulus for FM64
      modulus_bits = {45, 33, 45};
    }
  }

  parms.set_use_special_prime(false);
  parms.set_poly_modulus_degree(poly_deg);
  parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
  return parms;
}

}  // namespace

CheetorMulProt::CheetorMulProt(const std::shared_ptr<yacl::link::Context>& conn)
    : conn_(conn) {}

void CheetorMulProt::LazyInit(FieldType ft) {
  auto parms = DecideSEALParameters(ft);
  auto primes = GetWorkingPrimes(ft);
  const size_t nprimes = primes.size();

  // critical
  // FIXME(lwj): lock to thread-safe?

  if (!duplx_) {
    duplx_ = conn_->Spawn();
  }

  if (std::all_of(primes.begin(), primes.end(), [this](uint64_t prime) {
        return contexts_.count(prime) > 0;
      })) {
    return;
  }

  const int nxt_rank = conn_->NextRank();
  for (size_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx];
    if (contexts_.count(prime) > 0) {
      continue;
    }
    parms.set_plain_modulus(prime);
    seal::SEALContext this_context(parms, true, seal::sec_level_type::none);
    seal::KeyGenerator keygen(this_context);
    auto sk = std::make_shared<seal::SecretKey>(keygen.secret_key());
    auto pk = keygen.create_public_key();
    // NOTE(lwj): we patched seal/util/serializable.h
    auto pk_buf_send = cheetah::EncodeSEALObject(pk.obj());
    // exchange the public key
    yacl::Buffer pk_buf_recv;
    if (0 == nxt_rank) {
      conn_->Send(nxt_rank, pk_buf_send, "rank1 send pk");
      pk_buf_recv = conn_->Recv(nxt_rank, "rank1 recv pk");
    } else {
      pk_buf_recv = conn_->Recv(nxt_rank, "rank0 recv pk");
      conn_->Send(nxt_rank, pk_buf_send, "rank0 send pk");
    }
    auto peer_pub_key = std::make_shared<seal::PublicKey>();
    cheetah::DecodeSEALObject(pk_buf_recv, this_context, peer_pub_key.get());

    auto simd_prot = std::make_shared<spu::mpc::cheetah::SIMDMulProt>(
        parms.poly_modulus_degree(), prime);

    contexts_.insert({prime, this_context});
    secret_keys_.insert({prime, sk});
    peer_pub_keys_.insert({prime, peer_pub_key});
    simd_mul_instances_.insert({prime, simd_prot});
  }

  SPDLOG_INFO("CheetorMul uses N={}, modulus={} for input over {} bit ring",
              SIMDLane(ft), nprimes, SizeOf(ft) * 8);
}

NdArrayRef CheetorMulProt::MulToPrime(absl::Span<const uint64_t> x,
                                      FieldType ft) {
  std::vector<seal::Modulus> primes;
  for (uint64_t p : GetWorkingPrimes(ft)) {
    primes.emplace_back(p);
  }
  const int64_t nprimes = primes.size();
  SPU_ENFORCE(x.size() % nprimes == 0);
  const int64_t numel = x.size() / nprimes;

  // range check
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx].value();
    auto subspan = x.subspan(idx * numel, numel);
    SPU_ENFORCE(std::all_of(subspan.begin(), subspan.end(),
                            [&](uint64_t v) { return v < prime; }));
  }

  LazyInit(ft);

  const int rank = conn_->Rank();
  // Compute the share x * y mod p
  auto out_buff =
      std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
  auto mul_result = absl::MakeSpan(out_buff->data<uint64_t>(), numel * nprimes);

  // TODO(lwj): duplex Send & Recv for long input
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx].value();
    const auto& context = contexts_.find(prime)->second;
    auto simd_mul_prot = simd_mul_instances_.find(prime)->second;

    CheetorMulHelper helper(simd_mul_prot, context);

    if (rank == 0) {
      const auto& secret_key = secret_keys_.find(prime)->second;
      helper.MulPrimeShareSend(x.subspan(idx * numel, numel), *secret_key,
                               conn_, mul_result.subspan(idx * numel, numel));
    } else {
      const auto& pubkey = peer_pub_keys_.find(prime)->second;
      helper.MulPrimeShareRecv(x.subspan(idx * numel, numel), *pubkey, conn_,
                               mul_result.subspan(idx * numel, numel));
    }
  }

  auto stride = cheetah::CeilDiv<int64_t>(8 * nprimes, SizeOf(ft));
  return NdArrayRef(out_buff, makeType<PrimeShrTy>(ft), {numel}, {stride}, 0);
}

// Given x in [0, p0*p1*...*pn), compute x * x mod p0, mod p1, ..., mod pn
NdArrayRef CheetorMulProt::PrimeShareSquare(absl::Span<const uint64_t> x,
                                            FieldType ft) {
  std::vector<seal::Modulus> primes;
  for (uint64_t p : GetWorkingPrimes(ft)) {
    primes.emplace_back(p);
  }
  const int64_t nprimes = primes.size();
  SPU_ENFORCE(x.size() % nprimes == 0);
  const int64_t numel = x.size() / nprimes;

  // range check
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx].value();
    auto subspan = x.subspan(idx * numel, numel);
    SPU_ENFORCE(std::all_of(subspan.begin(), subspan.end(),
                            [&](uint64_t v) { return v < prime; }));
  }

  LazyInit(ft);

  const int rank = conn_->Rank();
  // Compute the share h0 * h1 mod p
  auto out_buff =
      std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
  auto mul_result = absl::MakeSpan(out_buff->data<uint64_t>(), numel * nprimes);

  // TODO(lwj): duplex Send & Recv for long input
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx].value();
    const auto& context = contexts_.find(prime)->second;
    auto simd_mul_prot = simd_mul_instances_.find(prime)->second;

    CheetorMulHelper helper(simd_mul_prot, context);

    if (rank == 0) {
      const auto& secret_key = secret_keys_.find(prime)->second;
      helper.MulPrimeShareSend(x.subspan(idx * numel, numel), *secret_key,
                               conn_, mul_result.subspan(idx * numel, numel));
    } else {
      const auto& pubkey = peer_pub_keys_.find(prime)->second;
      helper.MulPrimeShareRecv(x.subspan(idx * numel, numel), *pubkey, conn_,
                               mul_result.subspan(idx * numel, numel));
    }
  }

  // Square: hi = hi*hi + 2*<h0*h1> mod p
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    auto inp_slice = x.subspan(idx * numel, numel);
    auto mul_slice = mul_result.subspan(idx * numel, numel);

    const seal::Modulus& prime = primes[idx];
    // 2 * <h0*h1> mod p
    seal::util::add_poly_coeffmod(mul_slice.data(), mul_slice.data(), numel,
                                  prime, mul_slice.data());

    pforeach(0, numel, [&](int64_t i) {
      // += h[i] * h[i] via fused muladd
      mul_slice[i] = seal::util::multiply_add_uint_mod(
          inp_slice[i], inp_slice[i], mul_slice[i], prime);
    });
  }

  // NOTE(lwj): We keep the output in RNS form
  return NdArrayRef(out_buff, makeType<PrimeShrTy>(ft), {numel});
}

NdArrayRef CheetorMulProt::RingShareToPrimeShareSquare(const NdArrayRef& x) {
  SPU_ENFORCE(x.eltype().isa<RingTy>());
  auto ft = x.eltype().as<RingTy>()->field();
  LazyInit(ft);

  const int64_t numel = x.numel();
  const int rank = conn_->Rank();
  std::vector<seal::Modulus> primes;
  for (uint64_t p : GetWorkingPrimes(ft)) {
    primes.emplace_back(p);
  }
  const int64_t nprimes = primes.size();

  // NOTE(lwj): yacl::Buffer will not init the buffer which saves some time
  // for long buffer
  auto _hshr =
      std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
  auto hshr = absl::MakeSpan(_hshr->data<uint64_t>(), numel * nprimes);

  // local convert: h0 + h1 = x mod p
  for (size_t idx = 0; idx < primes.size(); ++idx) {
    ProbConvRing2kShareToPrimeShare(x, hshr.subspan(idx * numel, numel),
                                    primes[idx], rank);
  }

  NdArrayRef out_mod_p = PrimeShareSquare(hshr, ft);
  if (nprimes > 1) {
    SPU_ENFORCE_EQ(out_mod_p.elsize() * cheetah::calcNumel(out_mod_p.strides()),
                   sizeof(uint64_t) * nprimes);
    auto mul_result =
        absl::MakeSpan(out_mod_p.data<uint64_t>(), numel * nprimes);
    seal::util::RNSBase rns(primes, seal::MemoryManager::GetPool());
    // convert RNS to bigInt format
    rns.compose_array(mul_result.data(), numel, seal::MemoryManager::GetPool());
  }

  // NOTE(lwj): We store the RNS in u64. Thus for FM32 output, we
  // simply use stride=2 to wrap the underlying u64 buffer, and not to
  // allocate a new buffer.
  auto stride = cheetah::CeilDiv<int64_t>(8 * nprimes, SizeOf(ft));
  return NdArrayRef(out_mod_p.buf(), x.eltype(), {numel}, {stride},
                    /*offset*/ 0)
      .reshape(x.shape());
}

NdArrayRef CheetorMulProt::PrimeShareMul(absl::Span<const uint64_t> x_share,
                                         absl::Span<const uint64_t> y_share,
                                         FieldType ft) {
  std::vector<seal::Modulus> primes;
  for (uint64_t p : GetWorkingPrimes(ft)) {
    primes.emplace_back(p);
  }
  const int64_t nprimes = primes.size();
  SPU_ENFORCE(x_share.size() % nprimes == 0);
  const int64_t numel = x_share.size() / nprimes;

  // range check
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    uint64_t prime = primes[idx].value();
    auto subspan = x_share.subspan(idx * numel, numel);
    SPU_ENFORCE(std::all_of(subspan.begin(), subspan.end(),
                            [&](uint64_t v) { return v < prime; }));
  }

  LazyInit(ft);

  const int rank = conn_->Rank();

  auto mul_prog = [&, this](absl::Span<const uint64_t> hshr,
                            const std::shared_ptr<yacl::link::Context>& conn,
                            int sender_rank) {
    auto buff =
        std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
    auto cross_term = absl::MakeSpan(buff->data<uint64_t>(), numel * nprimes);
    for (int64_t idx = 0; idx < nprimes; ++idx) {
      uint64_t prime = primes[idx].value();
      const auto& context = contexts_.find(prime)->second;
      auto simd_mul_prot = simd_mul_instances_.find(prime)->second;

      CheetorMulHelper helper(simd_mul_prot, context);

      if (rank == sender_rank) {
        const auto& secret_key = secret_keys_.find(prime)->second;
        helper.MulPrimeShareSend(hshr.subspan(idx * numel, numel), *secret_key,
                                 conn, cross_term.subspan(idx * numel, numel));
      } else {
        const auto& pubkey = peer_pub_keys_.find(prime)->second;
        helper.MulPrimeShareRecv(hshr.subspan(idx * numel, numel), *pubkey,
                                 conn, cross_term.subspan(idx * numel, numel));
      }
    }
    return buff;
  };

  // compute x0*y1 and x1*y0 concurrently
  auto task0 = std::async(std::launch::async, mul_prog,
                          rank == 0 ? x_share : y_share, std::ref(conn_), 0);

  auto _cross1 = mul_prog(rank == 1 ? x_share : y_share, duplx_, 1);
  auto _cross0 = task0.get();
  auto cross0 = absl::MakeSpan(_cross0->data<uint64_t>(), numel * nprimes);
  auto cross1 = absl::MakeSpan(_cross1->data<uint64_t>(), numel * nprimes);

  // h0 = x0*y0 + <x0*y1> + <x1*y0> mod p
  // h1 = x1*y1 + <x0*y1> + <x1*y0> mod p
  for (int64_t idx = 0; idx < nprimes; ++idx) {
    auto xslice = x_share.subspan(idx * numel, numel);
    auto yslice = y_share.subspan(idx * numel, numel);
    auto c0slice = cross0.subspan(idx * numel, numel);
    auto c1slice = cross1.subspan(idx * numel, numel);

    // re-use the buffer
    auto out_slice = cross0.subspan(idx * numel, numel);
    const seal::Modulus& prime = primes[idx];
    pforeach(0, numel, [&](int64_t i) {
      out_slice[i] = seal::util::multiply_add_uint_mod(xslice[i], yslice[i],
                                                       c0slice[i], prime);
      out_slice[i] = seal::util::add_uint_mod(out_slice[i], c1slice[i], prime);
    });
  }

  auto stride = cheetah::CeilDiv<int64_t>(8 * nprimes, SizeOf(ft));
  auto otype = makeType<PrimeShrTy>(ft);
  return NdArrayRef(_cross0, otype, {numel}, {stride}, 0);
}

NdArrayRef CheetorMulProt::RingShareToPrimeShareMul(
    const NdArrayRef& x_ring_shr, const NdArrayRef& y_ring_shr) {
  SPU_ENFORCE(x_ring_shr.eltype().isa<RingTy>() and
              y_ring_shr.eltype().isa<RingTy>());
  SPU_ENFORCE_EQ(x_ring_shr.shape(), y_ring_shr.shape());
  auto ft = x_ring_shr.eltype().as<RingTy>()->field();
  LazyInit(ft);

  const int64_t numel = x_ring_shr.numel();
  const int rank = conn_->Rank();
  std::vector<seal::Modulus> primes;
  for (uint64_t p : GetWorkingPrimes(ft)) {
    primes.emplace_back(p);
  }
  const int64_t nprimes = primes.size();

  // NOTE(lwj): yacl::Buffer will not init the buffer which saves some time
  // for long buffer
  auto _x_prime_shr =
      std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
  auto _y_prime_shr =
      std::make_shared<yacl::Buffer>(numel * nprimes * sizeof(uint64_t));
  auto x_prime_shr =
      absl::MakeSpan(_x_prime_shr->data<uint64_t>(), numel * nprimes);
  auto y_prime_shr =
      absl::MakeSpan(_y_prime_shr->data<uint64_t>(), numel * nprimes);

  // local convert: h0 + h1 = x mod p
  for (size_t idx = 0; idx < primes.size(); ++idx) {
    ProbConvRing2kShareToPrimeShare(
        x_ring_shr, x_prime_shr.subspan(idx * numel, numel), primes[idx], rank);
    ProbConvRing2kShareToPrimeShare(
        y_ring_shr, y_prime_shr.subspan(idx * numel, numel), primes[idx], rank);
  }

  auto out_prime_shr = PrimeShareMul(x_prime_shr, y_prime_shr, ft);
  if (nprimes > 1) {
    // convert RNS to bigInt format
    SPU_ENFORCE_EQ(
        out_prime_shr.elsize() * cheetah::calcNumel(out_prime_shr.strides()),
        sizeof(uint64_t) * nprimes);
    auto u64_buff =
        absl::MakeSpan(out_prime_shr.data<uint64_t>(), numel * nprimes);
    seal::util::RNSBase rns(primes, seal::MemoryManager::GetPool());
    rns.compose_array(u64_buff.data(), numel, seal::MemoryManager::GetPool());
  }

  auto stride = cheetah::CeilDiv<int64_t>(8 * nprimes, SizeOf(ft));
  auto otype = makeType<PrimeShrTy>(ft);
  return NdArrayRef(out_prime_shr.buf(), otype, {numel}, {stride}, 0)
      .reshape(x_ring_shr.shape());
}

NdArrayRef CheetorMulProt::GetPrimesProduct(FieldType ft) {
  NdArrayRef out = ring_ones(ft, {1});

  auto num_primes = NumCRT(ft);
  DISPATCH_ALL_FIELDS(ft, "big_prime", [&]() {
    for (size_t i = 0; i < num_primes; ++i) {
      out.at<ring2k_t>(0) *= GetPlainPrime<ring2k_t>(i);
    }
  });
  return out;
}

std::vector<uint64_t> CheetorMulProt::GetWorkingPrimes(FieldType ft) {
  auto num_primes = NumCRT(ft);
  std::vector<uint64_t> out(num_primes);
  DISPATCH_ALL_FIELDS(ft, "primes", [&]() {
    for (size_t i = 0; i < num_primes; ++i) {
      out[i] = GetPlainPrime<ring2k_t>(i);
    }
  });
  return out;
}

size_t CheetorMulProt::PrimeNumBits(FieldType ft) {
  switch (ft) {
    case FM32:
      return 31;
    case FM64:
      return 60;
    case FM128:
      return 60 + 59;
    default:
      SPU_THROW("invalid ft={}", ft);
  }
}

CheetorMulProt::~CheetorMulProt() = default;

}  // namespace spu::mpc::cheetor
