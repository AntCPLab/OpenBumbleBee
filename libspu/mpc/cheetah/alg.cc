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

#include "libspu/mpc/cheetah/alg.h"

#include <array>
#include <future>

#include "conversion.h"

#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/state.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/util.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/tiled_dispatch.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

template <typename T>
static T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

static NdArrayRef batch_less_ap(KernelEvalContext* kctx, const NdArrayRef& x,
                                absl::Span<const float> y);

NdArrayRef batch_less_ap(KernelEvalContext* kctx, const NdArrayRef& x,
                         absl::Span<const float> y) {
  SPU_TRACE_MPC_LEAF(kctx->sctx());
  const int64_t numel = x.numel();
  const int64_t batch_size = y.size();

  const int fxp = kctx->sctx()->config().fxp_fraction_bits();
  const auto field = x.eltype().as<spu::Ring2k>()->field();
  const int rank = kctx->getState<Communicator>()->getRank();
  const int choice_provider = CompareProtocol::BatchedChoiceProvider();

  // CMP([x] > y) <=> MSB([x] - y) <=> 1{z0 > 2^{k-1} - z1}
  // Step 1: [x], y => [z] = [x - y]
  // x - y1, x - y2, ..., x - yB
  NdArrayRef msb_input;
  Shape ext_shape = x.shape();
  ext_shape.insert(ext_shape.end(), rank == choice_provider ? 1 : batch_size);

  if (rank != choice_provider) {
    msb_input = NdArrayRef(x.eltype(), ext_shape);
    DISPATCH_ALL_FIELDS(field, "subtract", [&]() {
      NdArrayView<ring2k_t> msb_inp(msb_input);
      for (int64_t b = 0; b < batch_size; ++b) {
        NdArrayView<const ring2k_t> _x(x);
        ring2k_t encoded_y =
            encodeToRing(PtBufferView(y[b]), field, fxp).at<ring2k_t>(0);

        pforeach(0, numel, [&](int64_t i) {
          msb_inp[i * batch_size + b] = _x[i] - encoded_y;
        });
      }
    });
  } else {
    msb_input = x.clone().reshape(ext_shape);
  }

  // Step 2: approximated MSB via dropping the low-end bits of each share
  int approx_less_prec = std::max(
      0, kctx->sctx()->config().cheetah_2pc_config().approx_less_precision());
  size_t bitwidth = SizeOf(field) * 8;

  if (approx_less_prec > 0 and approx_less_prec < fxp) {
    size_t bits_skip = fxp - approx_less_prec;
    ring_rshift_(msb_input, bits_skip);  // local truncate
    bitwidth -= bits_skip;
  }

  return DISPATCH_ALL_FIELDS(field, "cf2_mill", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = makeBitsMask<u2k>(bitwidth - 1);

    NdArrayView<const u2k> msb_inp(msb_input);
    NdArrayRef mill_input = ring_zeros(field, msb_input.shape());
    NdArrayView<u2k> mill_inp(mill_input);
    // Step 3: Compute MSB([z]) = msb(z0) ^ msb(z1) ^ 1{z0 > 2^{k - 1} - 1 - z1}
    if (rank != choice_provider) {
      SPU_ENFORCE_EQ(numel * batch_size, mill_input.numel());
      // z0
      pforeach(0, numel * batch_size,
               [&](int64_t i) { mill_inp[i] = msb_inp[i] & mask; });
    } else {
      SPU_ENFORCE_EQ(numel, mill_input.numel());
      // 2^{k - 1} - 1 - z1
      pforeach(0, numel,
               [&](int64_t i) { mill_inp[i] = (mask - msb_inp[i]) & mask; });
    }

    auto boolean_t = makeType<BShrTy>(field, 1);

    NdArrayRef out = cheetah::DispatchUnaryFuncWithBatchedInput(
        kctx, mill_input, rank != choice_provider, batch_size,
        [&](const NdArrayRef& inp,
            const std::shared_ptr<BasicOTProtocols>& base) {
          CompareProtocol comp_prot(base);

          int64_t numel = inp.numel();
          if (rank != choice_provider) {
            numel /= batch_size;
          }
          return comp_prot.BatchCompute(inp, true, numel, bitwidth, batch_size);
        });

    // Finally, local XOR the msb of the input share (i.e., msb(z0), msb(z1))
    if (rank != choice_provider) {
      NdArrayView<u2k> xcarry(out);
      pforeach(0, numel * batch_size, [&](int64_t i) {
        xcarry[i] ^= ((msb_inp[i] >> (bitwidth - 1)) & 1);
      });
    } else {
      NdArrayView<u2k> xcarry(out);
      pforeach(0, numel * batch_size, [&](int64_t i) {
        xcarry[i] ^= (((msb_inp[i / batch_size]) >> (bitwidth - 1)) & 1);
      });
    }

    return out.as(boolean_t);
  });
}

std::vector<NdArrayRef> BatchLessThan(KernelEvalContext* kctx,
                                      const NdArrayRef& x,
                                      absl::Span<const float> y) {
  SPU_ENFORCE(not y.empty());
  // x[0] < y[0], x[0] < y[1], ..., x[0] < y[B]

  auto batch_cmp = batch_less_ap(kctx, x, y);

  Index start_indices(batch_cmp.shape().size());
  Index end_indices(batch_cmp.shape().begin(), batch_cmp.shape().end());

  std::vector<NdArrayRef> out;
  int64_t batch = y.size();
  for (int64_t b = 0; b < batch; ++b) {
    start_indices.back() = b;
    end_indices.back() = b + 1;
    auto slice =
        batch_cmp.slice(start_indices, end_indices, {}).reshape(x.shape());
    out.emplace_back(slice);
  }
  return out;
}
}  // namespace spu::mpc::cheetah

namespace spu::mpc::cheetor {

NdArrayRef MulThenTrunc(KernelEvalContext* kctx, const NdArrayRef& x,
                        const NdArrayRef& y, FieldType working_ft, int fxp,
                        bool keep_ft) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>());
  SPU_ENFORCE(y.eltype().isa<Ring2k>());
  // SPU_ENFORCE(kctx->sctx()->config().cheetah_2pc_config().enable_cheetor());
  SPU_ENFORCE_EQ(x.shape(), y.shape());

  int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  int rank = kctx->lctx()->Rank();
  FieldType ft = x.eltype().as<Ring2k>()->field();
  if (working_ft == FT_INVALID) {
    working_ft = ft;
  }
  SPU_ENFORCE(SizeOf(ft) >= SizeOf(working_ft),
              "Can not cast from ring {} to prime {}", ft, working_ft);

  auto cheetor_mul_prot = kctx->getState<CheetorMulState>()->GetMulProt();

  // convert ring2k share to prime share
  std::vector<uint64_t> prime_modulus =
      CheetorMulProt::GetWorkingPrimes(working_ft);

  std::vector<uint64_t> x_prime_share(numel * prime_modulus.size());
  std::vector<uint64_t> y_prime_share(numel * prime_modulus.size());
  for (size_t i = 0; i < prime_modulus.size(); ++i) {
    ProbConvRing2kShareToPrimeShare(
        x, absl::MakeSpan(x_prime_share).subspan(i * numel, numel),
        prime_modulus[i], rank);
    ProbConvRing2kShareToPrimeShare(
        y, absl::MakeSpan(y_prime_share).subspan(i * numel, numel),
        prime_modulus[i], rank);
  }

  NdArrayRef muled;
  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_mul");
    muled = cheetor_mul_prot->PrimeShareMul(x_prime_share, y_prime_share,
                                            working_ft);
  }
  int64_t stride = sizeof(uint64_t) / SizeOf(working_ft);
  if (stride > 1) {
    // NOTE(lwj): for uint32 prime, we still store it using uint64
    // Thus, we set stride > 1 over uint64 array to represent uint32 array.
    muled = NdArrayRef(muled.buf(), muled.eltype(), muled.shape(), {stride}, 0);
  }

  if (prime_modulus.size() > 1) {
    // convert RNS form to bigint form
    std::vector<seal::Modulus> primes;
    for (uint64_t p : prime_modulus) {
      primes.emplace_back(p);
    }
    auto mul_result =
        absl::MakeSpan(muled.data<uint64_t>(), numel * prime_modulus.size());
    seal::util::RNSBase rns(primes, seal::MemoryManager::GetPool());
    // convert RNS to bigInt format
    rns.compose_array(mul_result.data(), numel, seal::MemoryManager::GetPool());
  }

  cheetor_mul_prot->LazyInit(working_ft);

  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_to_ring");
    auto out = cheetah::DispatchUnaryFunc(
                   kctx, muled,
                   [&](const NdArrayRef& input,
                       const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
                     PrimeRingCastProtocol::Meta meta;
                     meta.prime = CheetorMulProt::GetPrimesProduct(working_ft);
                     meta.prime_width =
                         CheetorMulProt::PrimeNumBits(working_ft);
                     meta.dst_ring = keep_ft ? ft : working_ft;
                     meta.dst_width = SizeOf(meta.dst_ring) * 8;
                     meta.truncate_nbits = fxp;

                     PrimeRingCastProtocol prime_cast_prot(base);
                     return prime_cast_prot.Compute(input, meta);
                   })
                   .as(makeType<cheetah::AShrTy>(keep_ft ? ft : working_ft))
                   .reshape(x.shape());
    return out;
  }
}

NdArrayRef SquareThenTrunc(KernelEvalContext* kctx, const NdArrayRef& x,
                           FieldType working_ft, int fxp, bool keep_ft) {
  SPU_ENFORCE(x.eltype().isa<Ring2k>());
  // SPU_ENFORCE(kctx->sctx()->config().cheetah_2pc_config().enable_cheetor());

  int64_t numel = x.numel();
  if (numel == 0) {
    return NdArrayRef(x.eltype(), x.shape());
  }

  int rank = kctx->lctx()->Rank();
  FieldType ft = x.eltype().as<Ring2k>()->field();
  if (working_ft == FT_INVALID) {
    working_ft = ft;
  }
  SPU_ENFORCE(SizeOf(ft) >= SizeOf(working_ft),
              "Can not cast from ring {} to prime {}", ft, working_ft);

  auto cheetor_mul_prot = kctx->getState<CheetorMulState>()->GetMulProt();

  // convert ring2k share to prime share
  std::vector<uint64_t> prime_modulus =
      CheetorMulProt::GetWorkingPrimes(working_ft);
  std::vector<uint64_t> prime_share(numel * prime_modulus.size());
  for (size_t i = 0; i < prime_modulus.size(); ++i) {
    ProbConvRing2kShareToPrimeShare(
        x, absl::MakeSpan(prime_share).subspan(i * numel, numel),
        prime_modulus[i], rank);
  }

  NdArrayRef square;
  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_square");
    square = cheetor_mul_prot->PrimeShareSquare(prime_share, working_ft);
  }

  int64_t stride = sizeof(uint64_t) / SizeOf(working_ft);
  if (stride > 1) {
    // NOTE(lwj): for uint32 prime, we still store it using uint64
    // Thus, we set stride > 1 over uint64 array to represent uint32 array.
    square =
        NdArrayRef(square.buf(), square.eltype(), square.shape(), {stride}, 0);
  }

  if (prime_modulus.size() > 1) {
    // convert RNS form to bigint form
    std::vector<seal::Modulus> primes;
    for (uint64_t p : prime_modulus) {
      primes.emplace_back(p);
    }
    auto mul_result =
        absl::MakeSpan(square.data<uint64_t>(), numel * prime_modulus.size());
    seal::util::RNSBase rns(primes, seal::MemoryManager::GetPool());
    // convert RNS to bigInt format
    rns.compose_array(mul_result.data(), numel, seal::MemoryManager::GetPool());
  }

  cheetor_mul_prot->LazyInit(working_ft);

  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_to_ring");
    auto out = cheetah::DispatchUnaryFunc(
                   kctx, square,
                   [&](const NdArrayRef& input,
                       const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
                     PrimeRingCastProtocol::Meta meta;
                     meta.prime = CheetorMulProt::GetPrimesProduct(working_ft);
                     meta.prime_width =
                         CheetorMulProt::PrimeNumBits(working_ft);
                     meta.dst_ring = keep_ft ? ft : working_ft;
                     meta.dst_width = SizeOf(meta.dst_ring) * 8;
                     meta.truncate_nbits = fxp;

                     PrimeRingCastProtocol prime_cast_prot(base);
                     return prime_cast_prot.Compute(input, meta);
                   })
                   .as(makeType<cheetah::AShrTy>(keep_ft ? ft : working_ft))
                   .reshape(x.shape());
    return out;
  }
}

NdArrayRef NExp_8(KernelEvalContext* kctx, const NdArrayRef& _x, int fxp) {
  SPU_ENFORCE(fxp > 0);
  const FieldType src_ft = _x.eltype().as<Ring2k>()->field();
  const FieldType working_ft = FM64;
  const int working_fxp = std::min(15, fxp);
  const int e_fxp = 12;  // 2^12 * log2(e) provides a precision of 4-digits

  // FM32 needs to lift-up to FM64
  // FM128 will down to FM64
  auto x = spu::mpc::cheetah::CastRing{}.proc(kctx, _x, working_ft,
                                              SignType::Negative);
  if (working_fxp < fxp) {
    SPU_ENFORCE((64 - 3) >= (fxp + 33),
                "fxp={} is too large, there might be a local truncation error",
                fxp);
    // local truncate. Error prob is bounded by 2^{fxp + 3 - 64}.
    // for x \in [-8, 0).
    spu::mpc::ring_arshift_(x, fxp - working_fxp);
  }

  // A + x * log2(e) >= 1 for x \in [-8, 0]
  // A >= 13 is enough.
  //
  // Multiply with log2(e) then add 13.
  const uint64_t E = std::roundf(std::log2(M_E) * (1L << e_fxp));
  const uint64_t A = 13ULL << (e_fxp + working_fxp);
  const int rank = kctx->lctx()->Rank();
  NdArrayView<uint64_t> xx(x);
  if (rank == 0) {
    pforeach(0, x.numel(), [&](int64_t i) {
      xx[i] *= E;
      xx[i] += A;
    });
  } else {
    pforeach(0, x.numel(), [&](int64_t i) { xx[i] *= E; });
  }

  // local truncate for |x| < 2^31 (i.e., -8*log2(e)*2^{15}*2^{12})
  // Thus, here the prob of truncation error is about 2^{-33} over FM64.
  spu::mpc::ring_arshift_(x, e_fxp);

  // int part is AShare over 2^{k - fxp}
  NdArrayRef _int_part(makeType<cheetor::AShrTy>(
                           working_ft, SizeOf(working_ft) * 8 - working_fxp),
                       x.shape());
  NdArrayView<uint64_t> int_part(_int_part);
  const uint64_t int_mask = (1UL << working_fxp) - 1;

  // Split the int-part
  pforeach(0, x.numel(),
           [&](int64_t i) { int_part[i] = xx[i] >> working_fxp; });

  seal::Modulus prime(CheetorMulProt::GetWorkingPrimes(working_ft)[0]);
  seal::Modulus prime_minus_one(prime.value() - 1);

  // convert from ring-share (int-part) to a prime share over p - 1
  ProbConvRing2kShareToPrimeShare(
      _int_part, absl::MakeSpan(_int_part.data<uint64_t>(), _int_part.numel()),
      prime_minus_one, rank);

  pforeach(0, x.numel(), [&](int64_t i) {
    // y = 2^int_part mod p
    uint64_t y = seal::util::exponentiate_uint_mod(2, int_part[i], prime);
    // z = 2^fract_part in RR
    float frac_part =
        static_cast<float>(xx[i] & int_mask) / (1L << working_fxp);
    frac_part = std::pow(2., frac_part);

    // Multiply the 2^{int_part} * 2^{frac_part} mod p
    int_part[i] = seal::util::multiply_uint_mod(
        y, static_cast<uint64_t>(std::roundf(frac_part * (1L << working_fxp))),
        prime);
  });

  auto cheetor_mul_prot = kctx->getState<CheetorMulState>()->GetMulProt();
  cheetor_mul_prot->LazyInit(working_ft);

  NdArrayRef muled;
  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_mul");
    muled = cheetor_mul_prot->MulToPrime(
        absl::MakeSpan(_int_part.data<uint64_t>(), _int_part.numel()),
        working_ft);
  }

  NdArrayRef out;
  {
    SPU_TRACE_ACTION(GET_TRACER(kctx), kctx->lctx(), (TR_MPC | TR_LAR),
                     (~TR_MPC), "prime_to_ring");
    out = cheetah::DispatchUnaryFunc(
        kctx, muled,
        [&](const NdArrayRef& inp,
            const std::shared_ptr<cheetah::BasicOTProtocols>& base) {
          PrimeRingCastProtocol cast(base);
          PrimeRingCastProtocol::Meta meta;
          meta.prime = CheetorMulProt::GetPrimesProduct(working_ft);
          meta.prime_width = CheetorMulProt::PrimeNumBits(working_ft);
          // For FM32, we need to cast down to FM64 first
          meta.dst_ring = src_ft == FM32 ? working_ft : src_ft;
          meta.dst_width = SizeOf(meta.dst_ring) * 8;
          meta.truncate_nbits = 13 + working_fxp - (fxp - working_fxp);
          return cast.Compute(inp, meta).as(
              makeType<cheetah::AShrTy>(meta.dst_ring));
        });
    out = out.reshape(x.shape());
  }

  if (src_ft != FM32) {
    return out;
  }

  return cheetah::CastRing{}.proc(kctx, out, src_ft, SignType::Positive);
}

}  // namespace spu::mpc::cheetor
