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

#include <unistd.h>

#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>

#include "gtest/gtest.h"
#include "seal/util/rns.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/parallel_utils.h"
#include "libspu/core/type.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/type.h"
#include "libspu/mpc/cheetah/dialect/cheetor/util.h"
#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetor::test {

class Seg3SigmoidTest : public ::testing::TestWithParam<FieldType> {
 public:
  static constexpr int64_t kBenchSize = 10000;
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetor, Seg3SigmoidTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<Seg3SigmoidTest::ParamType>& p) {
      return fmt::format("{}", p.param);
    });

template <typename T>
bool SignBit(T x) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<U>(x) >> (8 * sizeof(T) - 1) & 1;
}

void MaskItInplace(NdArrayRef a, size_t width) {
  auto field = a.eltype().as<Ring2k>()->field();
  if (width == SizeOf(field) * 8) {
    return;
  }
  DISPATCH_ALL_FIELDS(field, "mask", [&]() {
    NdArrayView<ring2k_t> _a(a);
    ring2k_t msk = (static_cast<ring2k_t>(1) << width) - 1;
    pforeach(0, _a.numel(), [&](int64_t i) { _a[i] &= msk; });
  });
}

// view [0, 2^k) as [-2^k/2, 2^k/2)
template <typename U>
auto ToSignType(U x, size_t width) {
  using S = typename std::make_signed<U>::type;
  if (sizeof(U) * 8 == width) {
    return static_cast<S>(x);
  }

  U half = static_cast<U>(1) << (width - 1);
  if (x >= half) {
    U upper = static_cast<U>(1) << width;
    x -= upper;
  }
  return static_cast<S>(x);
}

static NdArrayRef F_logistic(const NdArrayRef& x, uint32_t input_fxp, int rank,
                             cheetah::CheetahMul* mul_prot,
                             cheetah::RingExtendProtocol* ring_cast_prot,
                             cheetah::TruncateProtocol* ring_trunc_prot) {
  constexpr int approx_degree_ = 9;
  // double-angle for w1, w2, ..., w8
  int64_t num_approx_points = 2 * (approx_degree_ - 1);
  // approximate range x \in [-16, 16]
  double approx_interval_ = 16.;
  // sqrt(|w0|), sqrt(|w1|), ... sqrt(|w8|)
  // NOTE(juhou): we partition wj into both sine and cosine parts.
  // eg sqrt(|wj|)*sine(j) * sqrt(|wj|)*cosine(j) gives |wj|*sine(j)*cosine(j)
  // Aslo take care the sign.
  std::array<double, approx_degree_> sqrt_coeffs_ = {1.,
                                                     0.7856811722026087,
                                                     0.18492972212744474,
                                                     0.4115566184918635,
                                                     0.21455392047189895,
                                                     0.28578187418417367,
                                                     0.20820063862232982,
                                                     0.22518286726529987,
                                                     0.19226111755718367};

  auto field = x.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(SizeOf(field) * 8 >= 32, "Small ring is not supported");

  int64_t n = x.numel();
  uint32_t output_fxp = 29;                 // At most 31 - 2 bits
  uint32_t si_prec = output_fxp / 2;        // fxp for the sine part
  uint32_t cs_prec = output_fxp - si_prec;  // fxp for the cosine part

  // The input is fxp x*2^f for x \in [-16, 16]
  // So L = f + log(2 * 16)
  uint32_t period_nbits =
      static_cast<uint32_t>(std::log2(2. * approx_interval_)) + input_fxp;

  NdArrayRef _vole_input = ring_zeros(FM32, {n * num_approx_points});
  NdArrayView<uint32_t> vole_input(_vole_input);

  DISPATCH_ALL_FIELDS(field, "F_logistic_mul", [&]() {
    const double scaler = std::pow(2., period_nbits);
    auto mod_mask = (static_cast<ring2k_t>(1) << period_nbits) - 1;

    spu::NdArrayView<const ring2k_t> input(x);

    std::vector<double> local_real(n);
    pforeach(0, n, [&](int64_t i) {
      local_real[i] = static_cast<double>(input[i] & mod_mask) / scaler;
    });

    auto cast_to_ring = [&](double x) -> uint32_t {
      uint32_t ux = std::abs(x);
      if (std::signbit(x) and ux > 0) {
        ux = -ux;
      }
      return ux;
    };

    for (int64_t i = 0; i < n; ++i) {
      uint32_t* dst = &vole_input[i * num_approx_points];

      for (int k = 1, j = 0; k < approx_degree_; ++k, j += 2) {
        const double factor = 2. * M_PI * k;
        double sine = sqrt_coeffs_[k] * std::sin(local_real[i] * factor);
        double cosine = sqrt_coeffs_[k] * std::cos(local_real[i] * factor);

        double scaled_cosine = cosine * (1L << cs_prec);
        double scaled_sine = sine * (1L << si_prec);

        if (0 == rank) {
          // even term is negative
          scaled_cosine *= (k & 1 ? 1. : -1);
          scaled_sine *= (k & 1 ? 1. : -1);

          dst[j] = cast_to_ring(scaled_cosine);
          dst[j + 1] = cast_to_ring(scaled_sine);
        } else {
          dst[j + 1] = cast_to_ring(scaled_cosine);
          dst[j] = cast_to_ring(scaled_sine);
        }
      }
    }
  });

  // After the multiplication, fixed-point with output_fxp bits precision.
  // Compute x * y mod 2^32
  auto muled = mul_prot->MulOLE(_vole_input, rank == 0);
  auto sigmoid_u32 = ring_zeros(FM32, {n});

  NdArrayView<const uint32_t> muled_view(muled);
  NdArrayView<uint32_t> sigmoid(sigmoid_u32);
  seal::Modulus prime32(CheetorMulProt::GetWorkingPrimes(FM32)[0]);
  uint32_t init = rank > 0 ? 0 : sqrt_coeffs_[0] * 0.5 * (1L << output_fxp);
  for (int64_t i = 0; i < muled_view.numel(); i += num_approx_points) {
    // accumulate we perform lazy addmod here
    // \sum_j wj * sine(2*pi*j*x/2^L)
    uint32_t acc = init;
    for (int64_t j = 0; j < num_approx_points; ++j) {
      acc += muled_view[i + j];
    }
    sigmoid[i / num_approx_points] = acc;
  }

  if (field != FM32) {
    cheetah::RingExtendProtocol::Meta meta;
    meta.sign = SignType::Unknown;
    meta.use_heuristic = true;
    meta.signed_arith = true;
    meta.src_ring = FM32;
    meta.dst_ring = field;
    meta.src_width = SizeOf(FM32) * 8;
    meta.dst_width = SizeOf(field) * 8;
    auto out = ring_cast_prot->Compute(sigmoid_u32, meta).reshape(x.shape());
    // We can apply local truncate for the sigmoid output for larger ring
    ring_arshift_(out, output_fxp - input_fxp);
    return out;
  } else {
    cheetah::TruncateProtocol::Meta truc_meta;
    truc_meta.shift_bits = output_fxp - input_fxp;
    truc_meta.signed_arith = true;
    truc_meta.use_heuristic = true;
    return ring_trunc_prot->Compute(sigmoid_u32, truc_meta);
  }
}

static NdArrayRef F_logistic(const NdArrayRef& x, uint32_t input_fxp, int rank,
                             CheetorMulProt* mul_prot,
                             PrimeRingCastProtocol* prime2ring_prot) {
  constexpr int approx_degree_ = 9;
  // double-angle for w1, w2, ..., w8
  int64_t num_approx_points = 2 * (approx_degree_ - 1);
  // approximate range x \in [-16, 16]
  double approx_interval_ = 16.;
  // sqrt(|w0|), sqrt(|w1|), ... sqrt(|w8|)
  // NOTE(juhou): we partition wj into both sine and cosine parts.
  // eg sqrt(|wj|)*sine(j) * sqrt(|wj|)*cosine(j) gives |wj|*sine(j)*cosine(j)
  // Aslo take care the sign.
  std::array<double, approx_degree_> sqrt_coeffs_ = {1.,
                                                     0.7856811722026087,
                                                     0.18492972212744474,
                                                     0.4115566184918635,
                                                     0.21455392047189895,
                                                     0.28578187418417367,
                                                     0.20820063862232982,
                                                     0.22518286726529987,
                                                     0.19226111755718367};

  auto field = x.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(SizeOf(field) * 8 >= 32, "Small ring is not supported");

  int64_t n = x.numel();
  uint32_t output_fxp = 29;                 // At most 31 - 2 bits
  uint32_t si_prec = output_fxp / 2;        // fxp for the sine part
  uint32_t cs_prec = output_fxp - si_prec;  // fxp for the cosine part

  // The input is fxp x*2^f for x \in [-16, 16]
  // So L = f + log(2 * 16)
  uint32_t period_nbits =
      static_cast<uint32_t>(std::log2(2. * approx_interval_)) + input_fxp;

  std::vector<uint64_t> vole_input(n * num_approx_points);

  DISPATCH_ALL_FIELDS(field, "F_logistic_mul", [&]() {
    const double scaler = std::pow(2., period_nbits);
    auto mod_mask = (static_cast<ring2k_t>(1) << period_nbits) - 1;

    spu::NdArrayView<const ring2k_t> input(x);

    std::vector<double> local_real(n);
    pforeach(0, n, [&](int64_t i) {
      local_real[i] = static_cast<double>(input[i] & mod_mask) / scaler;
    });

    uint64_t prime32 = CheetorMulProt::GetWorkingPrimes(FM32)[0];
    auto cast_to_prime = [&](double x) -> uint64_t {
      uint64_t ux = std::abs(x);
      if (std::signbit(x) and ux > 0) {
        ux = prime32 - ux;
      }
      return ux;
    };

    for (int64_t i = 0; i < n; ++i) {
      uint64_t* dst = &vole_input[i * num_approx_points];

      for (int k = 1, j = 0; k < approx_degree_; ++k, j += 2) {
        const double factor = 2. * M_PI * k;
        double sine = sqrt_coeffs_[k] * std::sin(local_real[i] * factor);
        double cosine = sqrt_coeffs_[k] * std::cos(local_real[i] * factor);

        double scaled_cosine = cosine * (1L << cs_prec);
        double scaled_sine = sine * (1L << si_prec);

        if (0 == rank) {
          // even term is negative
          scaled_cosine *= (k & 1 ? 1. : -1);
          scaled_sine *= (k & 1 ? 1. : -1);

          dst[j] = cast_to_prime(scaled_cosine);
          dst[j + 1] = cast_to_prime(scaled_sine);
        } else {
          dst[j + 1] = cast_to_prime(scaled_cosine);
          dst[j] = cast_to_prime(scaled_sine);
        }
      }
    }
  });

  // After the multiplication, fixed-point with output_fxp bits precision.
  auto muled = mul_prot->MulToPrime(absl::MakeConstSpan(vole_input), FM32);
  auto sigmoid_u32 = ring_zeros(FM32, {n});

  NdArrayView<const uint32_t> muled_view(muled);
  NdArrayView<uint32_t> sigmoid(sigmoid_u32);
  seal::Modulus prime32(CheetorMulProt::GetWorkingPrimes(FM32)[0]);
  uint64_t init = rank > 0 ? 0 : sqrt_coeffs_[0] * 0.5 * (1L << output_fxp);
  for (int64_t i = 0; i < muled_view.numel(); i += num_approx_points) {
    // accumulate we perform lazy addmod here
    // \sum_j wj * sine(2*pi*j*x/2^L)
    uint64_t acc = init;
    for (int64_t j = 0; j < num_approx_points; ++j) {
      acc += muled_view[i + j];
    }
    sigmoid[i / num_approx_points] =
        seal::util::barrett_reduce_64(acc, prime32);
  }

  PrimeRingCastProtocol::Meta ring_cast_meta;
  ring_cast_meta.prime = CheetorMulProt::GetPrimesProduct(FM32);
  ring_cast_meta.prime_width = CheetorMulProt::PrimeNumBits(FM32);
  ring_cast_meta.dst_width = SizeOf(field) * 8;
  ring_cast_meta.dst_ring = field;
  ring_cast_meta.truncate_nbits = output_fxp - input_fxp;
  return prime2ring_prot->Compute(sigmoid_u32, ring_cast_meta)
      .reshape(x.shape());
}

TEST_P(Seg3SigmoidTest, FouriesSigmoid) {
  int64_t numel = kBenchSize;
  auto field = GetParam();
  NdArrayRef ring2k_shr[2];
  int msg_range = 3;
  int fxp_inp = 12;

  auto rnd_msg = ring_rand(field, {numel});
  // random message from [-2^{L + fxp}, 2^{L + fxp}]
  ring_lshift_(rnd_msg, SizeOf(field) * 8 - fxp_inp - msg_range);
  ring_rshift_(rnd_msg, SizeOf(field) * 8 - fxp_inp - msg_range);
  auto slice = rnd_msg.slice({0}, {numel}, {4});
  ring_neg_(slice);

  ring2k_shr[0] = ring_rand(field, rnd_msg.shape()).as(makeType<AShrTy>(field));
  ring2k_shr[1] = ring_sub(rnd_msg, ring2k_shr[0]).as(makeType<AShrTy>(field));

  size_t kWorldSize = 2;
  NdArrayRef old_outp[2];
  NdArrayRef outp[2];

  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    CheetorMulProt field_mul_prot(conn);
    cheetah::CheetahMul ring_mul_prot(conn, true);

    field_mul_prot.LazyInit(FM32);
    ring_mul_prot.LazyInitKeys(FM32);

    auto comm = std::make_shared<Communicator>(conn);
    auto ot_base = std::make_shared<cheetah::BasicOTProtocols>(
        comm, CheetahOtKind::YACL_Ferret);
    PrimeRingCastProtocol prime_prot(ot_base);
    cheetah::RingExtendProtocol ring_ext_prot(ot_base);
    cheetah::TruncateProtocol ring_trunc_prot(ot_base);

    size_t sent = conn->GetStats()->sent_bytes;
    outp[rank] = F_logistic(ring2k_shr[rank], fxp_inp, rank, &field_mul_prot,
                            &prime_prot);
    sent = conn->GetStats()->sent_bytes - sent;
    SPDLOG_INFO("Sigmoid (deg = 9) on n={} inputs ({}), sent {} bytes per",
                numel, field, sent * 1. / numel);

    sent = conn->GetStats()->sent_bytes;
    old_outp[rank] = F_logistic(ring2k_shr[rank], fxp_inp, rank, &ring_mul_prot,
                                &ring_ext_prot, &ring_trunc_prot);
    sent = conn->GetStats()->sent_bytes - sent;

    SPDLOG_INFO("Old Sigmoid (deg = 9) on n={} inputs ({}), sent {} bytes per",
                numel, field, sent * 1. / numel);
  });

  auto got = ring_add(outp[0], outp[1]);
  auto old_got = ring_add(old_outp[0], old_outp[1]);

  DISPATCH_ALL_FIELDS(field, "check", [&]() {
    using sT = std::make_signed<ring2k_t>::type;
    NdArrayView<sT> _msg(rnd_msg);
    NdArrayView<sT> _got(got);
    NdArrayView<sT> _oldgot(old_got);
    double scale = std::pow(2., fxp_inp);

    double max_err = 0.0;
    double old_max_err = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
      double x = _msg[i] / scale;
      double expected = 1. / (1. + std::exp(-x));
      double got = _got[i] / scale;
      double oldgot = _oldgot[i] / scale;
      max_err = std::max(max_err, std::abs(expected - got));
      old_max_err = std::max(old_max_err, std::abs(expected - oldgot));
    }
    printf("max error old: %f new: %f\n", old_max_err, max_err);
  });
}

}  // namespace spu::mpc::cheetor::test
