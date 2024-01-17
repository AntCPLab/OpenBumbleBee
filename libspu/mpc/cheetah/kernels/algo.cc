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

#include "libspu/mpc/cheetah/kernels/algo.h"

#include <array>
#include <future>

#include "libspu/core/encoding.h"
#include "libspu/core/trace.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/env.h"
#include "libspu/mpc/cheetah/kernels/truncate.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
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

class PowersCalculator {
 public:
  PowersCalculator(KernelEvalContext* kctx, bool is_fxp,
                   std::shared_ptr<yacl::link::Context> conn,
                   TruncA::OtConfig ot_config)
      : kctx_(kctx), conn_(conn), ot_config_(ot_config), is_fxp_(is_fxp) {
    mul_prot_ = kctx->getState<CheetahMulState>()->get()->getImpl();
    // NOTE(lwj): make sure the key is generated
    mul_prot_->Initialize(kctx->sctx()->getField());
  }

  std::array<NdArrayRef, 3> proc(const NdArrayRef& x, bool encryptor) {
    if (encryptor) {
      return actEncryptor(x);
    }

    return actEvaluator(x);
  }

  std::array<NdArrayRef, 3> actEncryptor(const NdArrayRef& x) {
    int64_t numel = x.numel();
    auto field = x.eltype().as<Ring2k>()->field();
    int nxt_rnk = conn_->NextRank();

    // Exchange Enc(x0) and Enc(x1)
    {
      auto enc_self_x = mul_prot_->EncryptArray(x);
      auto tag = fmt::format("send_x{}", conn_->Rank());
      for (auto& ct : enc_self_x) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), tag);
      }
    }
    auto enc_peer_x = mul_prot_->RecvEncryptedArray(field, numel, conn_.get());
    // Recv Enc(x0*x1)
    NdArrayRef square;
    {
      auto x0x1 = mul_prot_->RecvMulArray(field, numel, conn_.get());
      square = MayTruncate(ring_add(ring_mul(x, x), ring_add(x0x1, x0x1)),
                           SignType::Positive);
    }

    // Send Enc(s0)
    {
      auto enc_self_s = mul_prot_->EncryptArray(square);
      auto tag = fmt::format("send_s{}", conn_->Rank());
      for (auto& ct : enc_self_s) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), tag);
      }
    }
    // Recv Enc(s0*s1)
    NdArrayRef quad;
    {
      auto s0s1 = mul_prot_->RecvMulArray(field, numel, conn_.get());
      quad =
          MayTruncate(ring_add(ring_mul(square, square), ring_add(s0s1, s0s1)),
                      SignType::Positive);
    }

    NdArrayRef cubic;
    {
      auto iotask = std::async(
          [&]() { return mul_prot_->RecvMulArray(field, numel, conn_.get()); });

      std::vector<RLWECt>& enc_x1s0 = enc_peer_x;
      auto x1s0 = mul_prot_->MultiplyThenMask(square, absl::MakeSpan(enc_x1s0));
      auto x0s1 = iotask.get();
      for (auto& ct : enc_x1s0) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), "send_x1s0");
      }
      cubic = ring_mul(square, x);
      ring_add_(cubic, x0s1);
      ring_add_(cubic, x1s0);
      cubic = MayTruncate(cubic, SignType::Unknown);
    }

    return {square, cubic, quad};
  }

  std::array<NdArrayRef, 3> actEvaluator(const NdArrayRef& x) {
    int64_t numel = x.numel();
    auto field = x.eltype().as<Ring2k>()->field();
    int nxt_rnk = conn_->NextRank();
    // Exchange Enc(x0) and Enc(x1)
    auto enc_peer_x = mul_prot_->RecvEncryptedArray(field, numel, conn_.get());
    {
      auto enc_self_x = mul_prot_->EncryptArray(x);
      auto tag = fmt::format("send_x{}", conn_->Rank());
      for (auto& ct : enc_self_x) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), tag);
      }
    }

    // Send Enc(x0)*x1
    NdArrayRef square;
    {
      std::vector<RLWECt> enc_x0x1 = enc_peer_x;
      auto x0x1 = mul_prot_->MultiplyThenMask(x, absl::MakeSpan(enc_x0x1));
      for (auto& ct : enc_x0x1) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), "send_x1s0");
      }
      square = MayTruncate(ring_add(ring_mul(x, x), ring_add(x0x1, x0x1)),
                           SignType::Positive);
    }

    NdArrayRef quad;
    {
      auto enc_s0 = mul_prot_->RecvEncryptedArray(field, numel, conn_.get());
      auto s0s1 = mul_prot_->MultiplyThenMask(square, absl::MakeSpan(enc_s0));
      for (auto& ct : enc_s0) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), "send_s0s1");
      }
      quad =
          MayTruncate(ring_add(ring_mul(square, square), ring_add(s0s1, s0s1)),
                      SignType::Positive);
    }

    NdArrayRef cubic;
    {
      auto x1s0 =
          mul_prot_->MultiplyThenMask(square, absl::MakeSpan(enc_peer_x));
      for (auto& ct : enc_peer_x) {
        conn_->SendAsync(nxt_rnk, EncodeSEALObject(ct), "send_x1s0");
      }

      auto x0s1 = mul_prot_->RecvMulArray(field, numel, conn_.get());
      cubic = ring_mul(square, x);
      ring_add_(cubic, x0s1);
      ring_add_(cubic, x1s0);
      cubic = MayTruncate(cubic, SignType::Unknown);
    }

    return {square, cubic, quad};
  }

  NdArrayRef MayTruncate(const NdArrayRef& x, SignType s) {
    if (not is_fxp_) {
      return x;
    }

    TruncA trunc(ot_config_);
    return trunc.proc(kctx_, x, kctx_->sctx()->getFxpBits(), s);
  }

 private:
  KernelEvalContext* kctx_ = nullptr;
  std::shared_ptr<yacl::link::Context> conn_ = nullptr;
  CheetahMulImpl* mul_prot_ = nullptr;
  TruncA::OtConfig ot_config_;
  bool is_fxp_ = true;
};

static std::array<NdArrayRef, 3> f_powers_4(KernelEvalContext* kctx,
                                            const NdArrayRef& x, bool is_fxp) {
  SPU_TRACE_MPC_LEAF(kctx);
  const int64_t numel = x.numel();
  if (numel == 0) {
    return {NdArrayRef{x.eltype(), x.shape()},
            NdArrayRef{x.eltype(), x.shape()},
            NdArrayRef{x.eltype(), x.shape()}};
  }

  const size_t ot_instances = InitOTState(kctx, numel);
  const int64_t half = ot_instances == 1 ? numel : numel / 2;

  auto fx = x.reshape({x.numel()});
  TruncA::OtConfig config;
  config.instance_offset = 0;
  config.num_instances = half == numel ? ot_instances : ot_instances / 2;

  auto subtask = std::async([&]() {
    auto conn = kctx->getState<CheetahMulState>()->duplx();
    PowersCalculator pc(kctx, is_fxp, conn, config);

    auto x_slice = fx.slice({0}, {half}, {1});
    return pc.proc(x_slice, kctx->lctx()->Rank() == 0);
  });

  std::array<NdArrayRef, 3> tmp3;
  if (half != numel) {
    auto conn = kctx->getState<Communicator>()->lctx();
    auto cfg = config;
    cfg.instance_offset = config.num_instances;
    cfg.num_instances = ot_instances - config.num_instances;

    PowersCalculator pc(kctx, is_fxp, conn, cfg);
    auto x_slice = fx.slice({half}, {numel}, {1});
    tmp3 = pc.proc(x_slice, kctx->lctx()->Rank() != 0);
  }

  auto ret = subtask.get();
  if (half != numel) {
    std::array<NdArrayRef, 3> out;
    for (int i : {0, 1, 2}) {
      out[i] = NdArrayRef(x.eltype(), {numel});
      auto out_slice0 = out[i].slice({0}, {half}, {1});
      auto out_slice1 = out[i].slice({half}, {numel}, {1});
      std::memcpy(&out_slice0.at(0), &ret[i].at(0), x.elsize() * half);
      std::memcpy(&out_slice1.at(0), &tmp3[i].at(0),
                  x.elsize() * (numel - half));
      ret[i] = out[i];
    }
  }

  for (int i : {0, 1, 2}) {
    ret[i] = ret[i].reshape(x.shape());
  }
  return ret;
}

std::array<NdArrayRef, 3> ComputeUptoPower4(KernelEvalContext* kctx,
                                            const NdArrayRef& x, bool is_fxp) {
  return f_powers_4(kctx, x, is_fxp);
}

static NdArrayRef batch_less_ap(KernelEvalContext* kctx, const NdArrayRef& x,
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
  NdArrayRef msb_input;
  if (rank != choice_provider) {
    msb_input = NdArrayRef(x.eltype(), {numel * batch_size});
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
    msb_input = x.clone();
  }

  // Step 2: approximated MSB via dropping the low-end bits of each share
  constexpr size_t approx_tol_bits = 4;
  size_t bitwidth = SizeOf(field) * 8;
  if (TestEnvFlag(EnvFlag::SPU_CHEETAH_ENABLE_APPROX_LESS_THAN)) {
    size_t bits_skip = fxp - approx_tol_bits;
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
    NdArrayRef out(boolean_t, {numel * batch_size});
    const int64_t ot_ins = InitOTState(kctx, numel);
    const int64_t work_load_per = CeilDiv(numel, ot_ins);

    yacl::parallel_for(0, ot_ins, [&](int64_t bgn, int64_t end) {
      const size_t compare_radix = 4;
      auto ot_instance = kctx->getState<CheetahOTState>()->get(bgn);
      CompareProtocol comp_prot(ot_instance, compare_radix);

      for (int64_t job = bgn; job < end; ++job) {
        int64_t slice_bgn = std::min(numel, job * work_load_per);
        int64_t slice_end = std::min(numel, slice_bgn + work_load_per);
        int64_t slice_n = slice_end - slice_bgn;
        if (slice_n == 0) {
          break;
        }

        if (rank != choice_provider) {
          slice_bgn *= batch_size;
          slice_end *= batch_size;
        }

        // compute wrap bit
        auto out_slice = comp_prot.BatchCompute(
            mill_input.slice({slice_bgn}, {slice_end}, {1}),
            /*greater_than*/ true, slice_n, bitwidth, batch_size);

        if (rank != choice_provider) {
          std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                      out_slice.elsize() * out_slice.numel());
        } else {
          std::memcpy(&out.at(slice_bgn * batch_size), &out_slice.at(0),
                      out_slice.elsize() * out_slice.numel());
        }
      }
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
  auto fx = x.reshape({x.numel()});
  // x[0] < y[0], x[0] < y[1], ..., x[0] < y[B]
  auto batch_cmp = batch_less_ap(kctx, fx, y);

  std::vector<NdArrayRef> out;
  int64_t batch = y.size();
  for (int64_t b = 0; b < batch; ++b) {
    auto slice =
        batch_cmp.slice({b}, {batch_cmp.numel()}, {batch}).reshape(x.shape());
    out.emplace_back(slice);
  }
  return out;
}

NdArrayRef ChangeRing(KernelEvalContext* kctx, const NdArrayRef& x,
                      FieldType dst_field, SignType sign) {
  const auto src_field = x.eltype().as<spu::Ring2k>()->field();
  if (dst_field == src_field) {
    // nothing to do
    return x;
  }

  if (x.numel() == 0) {
    return NdArrayRef(makeType<AShrTy>(dst_field), {});
  }

  if (SizeOf(dst_field) < SizeOf(src_field)) {
    // given x0, x1 \in 2^k
    // return x0 mod 2^m, x1 mod 2^m
    NdArrayRef dst_x(makeType<AShrTy>(dst_field), {x.numel()});
    DISPATCH_ALL_FIELDS(src_field, "change_ring", [&]() {
      using T0 = ring2k_t;
      NdArrayView<const T0> src(x);
      DISPATCH_ALL_FIELDS(dst_field, "change_ring", [&]() {
        using T1 = ring2k_t;
        NdArrayView<T1> dst(dst_x);
        pforeach(0, x.numel(),
                 [&](int64_t i) { dst[i] = static_cast<T1>(src[i]); });
      });
    });
    return dst_x.reshape(x.shape());
  }

  // call 2PC protocol
  size_t n = x.numel();
  size_t num_job = InitOTState(kctx, n);
  size_t work_load = num_job == 0 ? 0 : CeilDiv(n, num_job);

  auto flatten_x = x.reshape({x.numel()});

  NdArrayRef out(makeType<AShrTy>(dst_field), {x.numel()});
  TiledDispatch(kctx, num_job, [&](int64_t job) {
    auto slice_bgn = std::min<int64_t>(n, job * work_load);
    auto slice_end = std::min<int64_t>(n, slice_bgn + work_load);
    if (slice_bgn == slice_end) {
      return;
    }

    auto ot_instance = kctx->getState<CheetahOTState>()->get(job);
    RingExtProtocol ext_prot(ot_instance);
    RingExtProtocol::Meta meta;
    meta.signed_arith = true;
    meta.use_heuristic = true;
    meta.src_field = src_field;
    meta.dst_field = dst_field;
    meta.sign = sign;

    auto out_slice =
        ext_prot.Compute(flatten_x.slice({slice_bgn}, {slice_end}, {1}), meta);
    std::memcpy(&out.at(slice_bgn), &out_slice.at(0),
                out_slice.elsize() * out_slice.numel());
  });

  return out.reshape(x.shape());
}

}  // namespace spu::mpc::cheetah
