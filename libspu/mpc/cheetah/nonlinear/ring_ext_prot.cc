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
#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"

#include "compare_prot.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

RingExtProtocol::RingExtProtocol(const std::shared_ptr<BasicOTProtocols>& base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
}

RingExtProtocol::~RingExtProtocol() { basic_ot_prot_->Flush(); }

NdArrayRef RingExtProtocol::ZeroExtend(const NdArrayRef& inp,
                                       const Meta& meta) {
  size_t src_width = SizeOf(meta.src_field) * 8;
  size_t dst_width = SizeOf(meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  NdArrayRef wrap = ComputeWrap(inp, meta);
  NdArrayRef out = ring_zeros(meta.dst_field, {inp.numel()});

  DISPATCH_ALL_FIELDS(meta.src_field, "zext", [&]() {
    using T0 = ring2k_t;
    NdArrayView<const T0> input(inp);
    DISPATCH_ALL_FIELDS(meta.dst_field, "zext", [&]() {
      using T1 = ring2k_t;
      NdArrayView<const T1> w(wrap);
      NdArrayView<T1> output(out);
      const T1 shift = static_cast<T1>(1) << src_width;
      const T1 mask = (static_cast<T1>(1) << (dst_width - src_width)) - 1;

      pforeach(0, input.numel(), [&](int64_t i) {
        output[i] = static_cast<T1>(input[i]) - shift * (w[i] & mask);
      });
    });
  });

  return out;
}

// Given msb(xA + xB mod 2^k) = 1, and xA, xB \in [0, 2^k)
// To compute w = 1{xA + xB > 2^{k} - 1}.
//            w = msb(xA) & msb(xB).
// COT msg corr=msb(xA) on choice msb(xB)
//    - msb(xB) = 0: get(-x, x) => 0
//    - msb(xB) = 1: get(-x, x + msb(xA)) => msb(xA)
NdArrayRef RingExtProtocol::MSB1ToWrap(const NdArrayRef& inp,
                                       const Meta& meta) {
  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const int wrap_width = meta.dst_ring_width - meta.src_ring_width;

  NdArrayRef cot_output = ring_zeros(meta.dst_field, inp.shape());
  DISPATCH_ALL_FIELDS(meta.src_field, "MSB1ToWrap", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;

    NdArrayView<const T0> xinp(inp);
    DISPATCH_ALL_FIELDS(meta.dst_field, "MSB1ToWrap", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      auto xout = absl::MakeSpan(&cot_output.at<T1>(0), cot_output.numel());

      if (rank == 0) {
        std::vector<T1> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = (xinp[i] >> (meta.src_ring_width - 1)) & 1;
        });
        auto sender = basic_ot_prot_->GetSenderCOT();
        sender->SendCAMCC(absl::MakeConstSpan(cot_input), xout, wrap_width);
        sender->Flush();
        pforeach(0, numel, [&](int64_t i) { xout[i] = -xout[i]; });
      } else {
        // choice bits
        std::vector<uint8_t> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = (xinp[i] >> (meta.src_ring_width - 1)) & 1;
        });
        basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(
            absl::MakeConstSpan(cot_input), xout, wrap_width);
      }
    });
  });

  return cot_output.as(makeType<BShrTy>(meta.dst_field, 1));
}

// Given msb(xA + xB mod 2^k) = 0, and xA, xB \in [0, 2^k)
// To compute w = 1{xA + xB > 2^{k} - 1}.
//
// Given msb(xA + xB mod 2^k) = 0
//   1. when xA + xB = x => w = 0
//   2. when xA + xB = x + 2^{k} => w = 1
//   For case 1: msb(xA) = msb(xB) = 0
//   For case 2: msb(xA) = 1 or msb(xB) = 1.
// Thus w = msb(xA) | msb(xB)
//
// 1-of-2 OT msg (r^msb(xA), r^1) on choice msb(xB)
//   - msb(xB) = 0: get (r, r^msb(xA)) => msb(xA)
//   - msb(xB) = 1: get (r, r^1) => 1
//
NdArrayRef RingExtProtocol::MSB0ToWrap(const NdArrayRef& inp,
                                       const Meta& meta) {
  const auto src_field = inp.eltype().as<Ring2k>()->field();
  const size_t src_width = SizeOf(src_field) * 8;
  const size_t dst_width = SizeOf(meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  const int64_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;

  if (0 == rank) {
    wrap_bool = ring_randbit(meta.src_field, {numel});
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(src_field, "msb0", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const T0> xinp(inp);
      NdArrayView<const T0> xrnd(wrap_bool);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      pforeach(0, numel, [&](int64_t i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (src_width - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      });
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);

    DISPATCH_ALL_FIELDS(src_field, "msb0", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const T0> xinp(inp);
      pforeach(0, numel, [&](int64_t i) {
        choices[i] = (xinp[i] >> (src_width - 1)) & 1;
      });

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      wrap_bool = ring_zeros(meta.src_field, {numel});
      NdArrayView<T0> xoup(wrap_bool);
      pforeach(0, numel,
               [&](int64_t i) { xoup[i] = static_cast<T0>(recv[i] & 1); });
    });
  }

  NdArrayRef ext_wrap = ring_zeros(meta.dst_field, {inp.numel()});
  DISPATCH_ALL_FIELDS(src_field, "wrap.adj", [&]() {
    using T0 = ring2k_t;
    NdArrayView<const T0> w(wrap_bool);
    DISPATCH_ALL_FIELDS(meta.dst_field, "wrap.adj", [&]() {
      using T1 = ring2k_t;
      NdArrayView<T1> ext_w(ext_wrap);
      pforeach(0, inp.numel(),
               [&](int64_t i) { ext_w[i] = static_cast<T1>(w[i]); });
    });
  });

  return basic_ot_prot_->B2ASingleBitWithSize(
      ext_wrap.as(makeType<BShrTy>(meta.dst_field, 1)), dst_width - src_width);
}

NdArrayRef RingExtProtocol::Compute(const NdArrayRef& inp, Meta _meta) {
  size_t src_width = SizeOf(_meta.src_field) * 8;
  size_t dst_width = SizeOf(_meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  if (_meta.src_ring_width == 0) {
    _meta.src_ring_width = src_width;
  }
  if (_meta.dst_ring_width == 0) {
    _meta.dst_ring_width = dst_width;
  }

  const int rank = basic_ot_prot_->Rank();

  if (_meta.signed_arith && _meta.sign == SignType::Unknown &&
      _meta.use_heuristic) {
    Meta meta = _meta;
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    meta.use_heuristic = false;
    meta.sign = SignType::Positive;
    meta.signed_arith = false;

    if (rank == 0) {
      NdArrayRef tmp = inp.clone();
      DISPATCH_ALL_FIELDS(_meta.src_field, "ext.adj", [&] {
        NdArrayView<ring2k_t> _inp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (src_width - kHeuristicBound);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _inp[i] = _inp[i] + big_value; });
      });

      tmp = Compute(tmp, meta);

      DISPATCH_ALL_FIELDS(meta.dst_field, "ext.adj", [&] {
        NdArrayView<ring2k_t> _outp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (src_width - kHeuristicBound);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _outp[i] = _outp[i] - big_value; });
      });
      return tmp;
    } else {
      return Compute(inp, meta);
    }
  }

  Meta meta = _meta;

  if (_meta.signed_arith && _meta.sign != SignType::Unknown) {
    meta.sign = _meta.sign == SignType::Positive ? SignType::Negative
                                                 : SignType::Positive;
  }

  NdArrayRef out = DISPATCH_ALL_FIELDS(meta.src_field, "ext.zext", [&]() {
    using T0 = ring2k_t;
    NdArrayView<const ring2k_t> xinp(inp);

    if (meta.signed_arith && rank == 0) {
      const T0 component = (static_cast<T0>(1) << (src_width - 1));
      // SExt(x, m, n) = ZExt(x + 2^{m-1} mod 2^m, m, n) - 2^{m-1}
      auto tmp = ring_zeros(meta.src_field, {inp.numel()});
      NdArrayView<T0> xtmp(tmp);
      pforeach(0, inp.numel(),
               [&](int64_t i) { xtmp[i] = xinp[i] + component; });
      return ZeroExtend(tmp, meta);
    } else {
      return ZeroExtend(inp, meta);
    }
  });

  if (meta.signed_arith && rank == 0) {
    DISPATCH_ALL_FIELDS(meta.dst_field, "ext.adj", [&]() {
      const auto component = (static_cast<ring2k_t>(1) << (src_width - 1));
      // SExt(x, m, n) = ZExt(x + 2^{m-1} mod 2^m, m, n) - 2^{m-1}
      NdArrayView<ring2k_t> xout(out);
      pforeach(0, inp.numel(), [&](int64_t i) { xout[i] -= component; });
    });
  }

  return out;
}

NdArrayRef RingExtProtocol::ComputeWrap(const NdArrayRef& inp,
                                        const Meta& meta) {
  const int rank = basic_ot_prot_->Rank();
  const auto src_field = inp.eltype().as<Ring2k>()->field();

  switch (meta.sign) {
    case SignType::Positive:
      return MSB0ToWrap(inp, meta);
    case SignType::Negative:
      return MSB1ToWrap(inp, meta);
    case SignType::Unknown:
    default:
      break;
  }

  CompareProtocol compare_prot(basic_ot_prot_);
  NdArrayRef wrap_bool;
  // w = 1{x_A + x_B > 2^k - 1}
  //   = 1{x_A > 2^k - 1 - x_B}
  if (rank == 0) {
    wrap_bool = compare_prot.Compute(inp, true);
  } else {
    auto adjusted = ring_neg(inp);
    DISPATCH_ALL_FIELDS(src_field, "wrap.adj", [&]() {
      NdArrayView<ring2k_t> xadj(adjusted);
      pforeach(0, inp.numel(), [&](int64_t i) { xadj[i] -= 1; });
    });
    wrap_bool = compare_prot.Compute(adjusted, true);
  }

  NdArrayRef ext_wrap = ring_zeros(meta.dst_field, {inp.numel()});
  DISPATCH_ALL_FIELDS(src_field, "wrap.adj", [&]() {
    using T0 = ring2k_t;
    NdArrayView<const T0> w(wrap_bool);
    DISPATCH_ALL_FIELDS(meta.dst_field, "wrap.adj", [&]() {
      using T1 = ring2k_t;
      NdArrayView<T1> ext_w(ext_wrap);
      pforeach(0, inp.numel(),
               [&](int64_t i) { ext_w[i] = static_cast<T1>(w[i]); });
    });
  });

  return basic_ot_prot_->B2ASingleBitWithSize(
      ext_wrap.as(makeType<BShrTy>(meta.dst_field, 1)),
      meta.dst_ring_width - meta.src_ring_width);
}

// Given h0 + h1 = h mod p and h < p / 2
// Define b0 = 1{h0 >= p/2}
//        b1 = 1{h1 >= p/2}
// Define w = 1{h0 + h1 >= p}
//
// We have w = b0 | b1
// 1-of-2 OT msg (r^b0, r^1) on choice b1
//   - b1 = 0: get (r, r^b0) => bshare of b0
//   - b1 = 1: get (r, r^1) => bshare of 1
[[maybe_unused]] static NdArrayRef MSB0ToWrapPrime(
    const NdArrayRef& inp, FieldType src_field, const NdArrayRef& prime,
    FieldType dst_field, const std::shared_ptr<BasicOTProtocols>& base_ot) {
  const int64_t numel = inp.numel();
  const int rank = base_ot->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  NdArrayRef wrap_bool;

  if (0 == rank) {
    wrap_bool = ring_randbit(src_field, {numel});
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(src_field, "msb0", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      T0 phalf = NdArrayView<ring2k_t>(prime)[0] / 2;
      NdArrayView<const T0> xinp(inp);
      NdArrayView<const T0> xrnd(wrap_bool);
      pforeach(0, numel, [&](int64_t i) {
        send[2 * i + 0] = xrnd[i] ^ static_cast<uint8_t>(xinp[i] >= phalf);
        send[2 * i + 1] = xrnd[i] ^ 1;
      });
    });

    auto sender = base_ot->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);

    DISPATCH_ALL_FIELDS(src_field, "msb0", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      T0 phalf = NdArrayView<ring2k_t>(prime)[0] / 2;
      NdArrayView<const T0> xinp(inp);
      pforeach(0, numel, [&](int64_t i) {
        choices[i] = static_cast<uint8_t>(xinp[i] >= phalf);
      });

      std::vector<uint8_t> recv(numel);
      base_ot->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                          absl::MakeSpan(recv), nbits);

      wrap_bool = ring_zeros(src_field, {numel});
      NdArrayView<T0> xoup(wrap_bool);
      pforeach(0, numel,
               [&](int64_t i) { xoup[i] = static_cast<T0>(recv[i] & 1); });
    });
  }

  NdArrayRef ext_wrap = ring_zeros(dst_field, {inp.numel()});
  DISPATCH_ALL_FIELDS(src_field, "wrap.adj", [&]() {
    using T0 = ring2k_t;
    NdArrayView<const T0> w(wrap_bool);
    DISPATCH_ALL_FIELDS(dst_field, "wrap.adj", [&]() {
      using T1 = ring2k_t;
      NdArrayView<T1> ext_w(ext_wrap);
      pforeach(0, inp.numel(),
               [&](int64_t i) { ext_w[i] = static_cast<T1>(w[i]); });
    });
  });

  // size_t bw = absl::bit_width(prime) - 1;
  return base_ot->B2ASingleBitWithSize(
      ext_wrap.as(makeType<BShrTy>(dst_field, 1)), SizeOf(dst_field) * 8);
}

NdArrayRef RingExtPrime(const NdArrayRef& input, const NdArrayRef& prime,
                        FieldType dst_field, std::optional<size_t> trunc_bits,
                        const std::shared_ptr<BasicOTProtocols>& base_ot) {
  FieldType src_field = input.eltype().as<Ring2k>()->field();

  DISPATCH_ALL_FIELDS(src_field, "check_range", [&]() {
    NdArrayView<ring2k_t> _input(input);
    ring2k_t _prime = NdArrayView<ring2k_t>(prime)[0];
    for (int64_t i = 0; i < _input.numel(); ++i) {
      SPU_ENFORCE(_input[i] < _prime, "invalid share {}", _input[i]);
    }
  });

  const int64_t numel = input.numel();
  const int rank = base_ot->Rank();
  const size_t shft = trunc_bits ? *trunc_bits : 0;
  SPU_ENFORCE(shft < 8 * SizeOf(dst_field), "truncate too much {}", shft);

  // w = 1{h0 + h1 > p - 1}
  // |h| < p / 2 => 0 < h + p/2 < p
  NdArrayRef output = ring_zeros(dst_field, input.shape());
  if (rank == 0) {
    auto wrap_arith =
        MSB0ToWrapPrime(input, src_field, prime, dst_field, base_ot);
    DISPATCH_ALL_FIELDS(src_field, "finalize", [&]() {
      using T0 = ring2k_t;
      T0 _prime = NdArrayView<T0>(prime)[0];
      NdArrayView<const T0> _input(input);
      DISPATCH_ALL_FIELDS(dst_field, "finalize", [&]() {
        using T1 = ring2k_t;
        NdArrayView<T1> _output(output);
        NdArrayView<T1> _wrap(wrap_arith);

        pforeach(0, numel, [&](int64_t i) {
          _output[i] = static_cast<T1>(_input[i] >> shft) -
                       static_cast<T1>(_prime >> shft) * _wrap[i];
        });
      });
    });
    return output.as(makeType<AShrTy>(dst_field));
  }

  /// rank = 1
  auto adjusted = input.clone();
  DISPATCH_ALL_FIELDS(src_field, "wrap.adj", [&]() {
    using T0 = ring2k_t;
    ring2k_t _prime = NdArrayView<T0>(prime)[0];
    // FIXME(lwj): can we use absl::bit_width(uint128) ?
    size_t prime_bw = std::ceil(std::log2(1. * _prime));
    NdArrayView<ring2k_t> xadj(adjusted);
    ring2k_t big_positive = static_cast<ring2k_t>(1) << (prime_bw - 2);
    pforeach(0, numel, [&](int64_t i) {
      xadj[i] = big_positive + xadj[i];
      xadj[i] -= (xadj[i] >= _prime ? _prime : 0);
    });
  });
  auto wrap_arith =
      MSB0ToWrapPrime(adjusted, src_field, prime, dst_field, base_ot);

  DISPATCH_ALL_FIELDS(src_field, "finalize", [&]() {
    using T0 = ring2k_t;
    T0 _prime = NdArrayView<T0>(prime)[0];
    size_t prime_bw = std::ceil(std::log2(1. * _prime));
    NdArrayView<const T0> _input(adjusted);
    DISPATCH_ALL_FIELDS(dst_field, "finalize", [&]() {
      using T1 = ring2k_t;
      NdArrayView<T1> _output(output);
      NdArrayView<T1> _wrap(wrap_arith);
      ring2k_t big_positive = static_cast<ring2k_t>(1) << (prime_bw - 2 - shft);

      pforeach(0, numel, [&](int64_t i) {
        _output[i] = static_cast<T1>(_input[i] >> shft) -
                     static_cast<T1>(_prime >> shft) * _wrap[i] - big_positive;
      });
    });
  });

  return output.as(makeType<AShrTy>(dst_field));
}

}  // namespace spu::mpc::cheetah
