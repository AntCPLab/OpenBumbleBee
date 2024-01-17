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

#include "libspu/mpc/bumblebee/nonlinear/compare_prot.h"

#include "yacl/crypto/tools/prg.h"
#include "yacl/link/link.h"

#include "libspu/core/type.h"
#include "libspu/mpc/bumblebee/ot/basic_ot_prot.h"
#include "libspu/mpc/bumblebee/ot/util.h"
#include "libspu/mpc/bumblebee/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::bumblebee {

CompareProtocol::CompareProtocol(std::shared_ptr<BasicOTProtocols> base,
                                 size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;
}

CompareProtocol::~CompareProtocol() { basic_ot_prot_->Flush(); }

void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                  uint8_t rnd_cmp_bit, uint8_t rnd_eq_bit, bool gt,
                  size_t batch_shift = 0) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, "N={} got digit={}", N, digit);
  for (size_t i = 0; i < N; i++) {
    uint8_t eq_cmp = 0;
    if (gt) {
      eq_cmp = rnd_cmp_bit ^ static_cast<uint8_t>(digit > i);
    } else {
      eq_cmp = rnd_cmp_bit ^ static_cast<uint8_t>(digit < i);
    }
    // compact two bits into one OT message
    eq_cmp |= ((rnd_eq_bit ^ static_cast<uint8_t>(digit == i)) << 1);

    // compact batch eq and cmp bits into one OT message
    // NOTE(lwj) Now only support Batch size <= 4
    ot_messages[i] |= (eq_cmp << batch_shift);
  }
}

// The Mill protocol from "CrypTFlow2: Practical 2-Party Secure Inference"
// Algorithm 1. REF: https://arxiv.org/pdf/2010.06457.pdf
NdArrayRef CompareProtocol::DoCompute(const NdArrayRef& inp, bool greater_than,
                                      NdArrayRef* keep_eq, int64_t bitwidth) {
  SPU_ENFORCE(inp.shape().size() == 1, "need 1D array");
  auto field = inp.eltype().as<Ring2k>()->field();
  // align to multiple of compare radix
  if ((bitwidth % compare_radix_) != 0) {
    bitwidth = CeilDiv<int64_t>(bitwidth, compare_radix_) * compare_radix_;
  }
  int64_t num_digits = CeilDiv(bitwidth, (int64_t)compare_radix_);
  size_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  int64_t num_cmp = inp.numel();
  // init to all zero
  std::vector<uint8_t> digits(num_cmp * num_digits, 0);

  // Step 1 break into digits \in [0, radix)
  DISPATCH_ALL_FIELDS(field, "break_digits", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < num_cmp; ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
      }
    }
  });

  std::vector<uint8_t> leaf_cmp(num_cmp * num_digits, 0);
  std::vector<uint8_t> leaf_eq(num_cmp * num_digits, 0);
  if (is_sender_) {
    // Step 2 sample random bits
    yacl::crypto::Prg<uint8_t> prg;
    prg.Fill(absl::MakeSpan(leaf_cmp));
    prg.Fill(absl::MakeSpan(leaf_eq));

    // convert u8 random to boolean random
    std::transform(leaf_cmp.begin(), leaf_cmp.end(), leaf_cmp.data(),
                   [](uint8_t v) { return v & 1; });
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return v & 1; });

    // Step 6-7 set the OT messages with two packed bits (one for compare, one
    // for equal)
    std::vector<uint8_t> leaf_ot_msg(radix * num_cmp * num_digits, 0);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_cmp * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] =
          absl::Span<uint8_t>{leaf_ot_msg.data() + i * radix, radix};
    }

    for (int64_t i = 0; i < num_cmp; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_cmp = leaf_cmp.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      for (int64_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_cmp = this_leaf_cmp[j] & 1;
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_cmp, rnd_eq,
                     greater_than);
      }
    }

    // Step 9: sender of n*M instances of 1-of-N OT
    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 2);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    // Step 10: receiver of 1-of-N OT
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits), radix,
                                               absl::MakeSpan(leaf_cmp), 2);
    // extract equality bits from packed messages
    for (int64_t i = 0; i < num_cmp; ++i) {
      auto* this_leaf_cmp = leaf_cmp.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;
      for (int64_t j = 0; j < num_digits; ++j) {
        this_leaf_eq[j] = (this_leaf_cmp[j] >> 1) & 1;
        this_leaf_cmp[j] &= 1;
      }
    }
  }

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_cmp =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
          .as(boolean_t);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
          .as(boolean_t);

  DISPATCH_ALL_FIELDS(field, "copy_leaf", [&]() {
    NdArrayView<ring2k_t> xprev_cmp(prev_cmp);
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, prev_cmp.numel(), [&](int64_t i) {
      xprev_cmp[i] = leaf_cmp[i];
      xprev_eq[i] = leaf_eq[i];
    });
  });

  // Step 12 - 17. Evaluate the traversal AND
  if (keep_eq == nullptr) {
    // Optimization when keep_eq is false
    // Section 3.1.1: "Removing unnecessary equality computations"
    return TraversalAND(prev_cmp, prev_eq, num_cmp, num_digits).as(boolean_t);
  }

  auto [_gt, _eq] = TraversalANDWithEq(prev_cmp, prev_eq, num_cmp, num_digits);
  *keep_eq = _eq;
  keep_eq->as(boolean_t);
  return _gt.as(boolean_t);
}

NdArrayRef CompareProtocol::DoBatchCompute(const NdArrayRef& inp,
                                           bool greater_than, int64_t numelt,
                                           int64_t bitwidth,
                                           int64_t batch_size) {
  auto field = inp.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(batch_size);
  SPU_ENFORCE(bitwidth > 0 && bitwidth <= (int)SizeOf(field) * 8);

  if (bitwidth % compare_radix_ != 0) {
    bitwidth = CeilDiv<int64_t>(bitwidth, compare_radix_) * compare_radix_;
  }

  int64_t num_digits = CeilDiv<int64_t>(bitwidth, compare_radix_);
  size_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  size_t num_cmp = numelt * batch_size;
  // init to all zero
  std::vector<uint8_t> digits(inp.numel() * num_digits, 0);

  // Step 1 break into digits \in [0, radix)
  DISPATCH_ALL_FIELDS(field, "break_digits", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < inp.numel(); ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
      }
    }
  });

  std::vector<uint8_t> leaf_cmp(num_cmp * num_digits, 0);
  std::vector<uint8_t> leaf_eq(num_cmp * num_digits, 0);
  if (is_sender_) {
    // NOTE(lwj): sender holds the batched message and receiver inputs the
    // choice bits
    // Step 2 sample random bits
    yacl::crypto::Prg<uint8_t> prg;
    prg.Fill(absl::MakeSpan(leaf_cmp));
    prg.Fill(absl::MakeSpan(leaf_eq));
    // convert to boolean
    std::transform(leaf_cmp.begin(), leaf_cmp.end(), leaf_cmp.data(),
                   [](uint8_t v) { return v & 1; });
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return v & 1; });

    // Step 6-7 set the OT messages with two packed bits (one for compare, one
    // for equal)
    std::vector<uint8_t> _leaf_ot_msg(radix * numelt * num_digits, 0);
    auto leaf_ot_msg = absl::MakeSpan(_leaf_ot_msg);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(numelt * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] = leaf_ot_msg.subspan(i * radix, radix);
    }

    for (int64_t i = 0; i < numelt; ++i) {
      for (int64_t j = 0; j < batch_size; ++j) {
        auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
        auto* this_digit = digits.data() + (i * batch_size + j) * num_digits;
        auto* this_leaf_cmp =
            leaf_cmp.data() + (i * batch_size + j) * num_digits;
        auto* this_leaf_eq = leaf_eq.data() + (i * batch_size + j) * num_digits;

        for (int64_t k = 0; k < num_digits; ++k) {
          uint8_t rnd_cmp = this_leaf_cmp[k];
          uint8_t rnd_eq = this_leaf_eq[k];
          SetLeafOTMsg(this_ot_msg[k], this_digit[k], rnd_cmp, rnd_eq,
                       greater_than, /*offset*/ j * 2);
        }
      }
    }

    // Step 9: sender of n*M instances of 1-of-N OT
    basic_ot_prot_->GetSenderCOT()->SendCMCC(leaf_ot_msg, radix,
                                             /*ot_bitwidth*/ batch_size * 2);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    // Step 10: receiver of 1-of-N OT
    std::vector<uint8_t> packed_cmp(numelt * num_digits, 0);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(
        absl::MakeConstSpan(digits), radix, absl::MakeSpan(packed_cmp),
        /*bitwidth*/ batch_size * 2);

    // extract equality bits from packed messages
    // See `SetLeafOTMsg`
    // Need to unpack the cmp array from batch
    // eq_n||cmp_n||eq_n-1||cmp_n-1||...||eq0||cmp0
    for (int64_t i = 0; i < numelt; ++i) {
      auto* this_pack_cmp = packed_cmp.data() + i * num_digits;
      for (int64_t j = 0; j < batch_size; ++j) {
        auto* this_leaf_cmp =
            leaf_cmp.data() + (i * batch_size + j) * num_digits;
        auto* this_leaf_eq = leaf_eq.data() + (i * batch_size + j) * num_digits;
        for (int64_t k = 0; k < num_digits; ++k) {
          this_leaf_eq[k] = (this_pack_cmp[k] >> (j * 2 + 1)) & 1;
          this_leaf_cmp[k] = (this_pack_cmp[k] >> (j * 2)) & 1;
        }
      }
    }
  }

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_cmp =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
          .as(boolean_t);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
          .as(boolean_t);

  DISPATCH_ALL_FIELDS(field, "extract_ot_msg", [&]() {
    NdArrayView<ring2k_t> xprev_cmp(prev_cmp);
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, xprev_cmp.numel(), [&](int64_t i) {
      xprev_cmp[i] = leaf_cmp[i];
      xprev_eq[i] = leaf_eq[i];
    });
  });

  // Step 12 - 17. Evaluate the traversal AND
  return TraversalAND(prev_cmp, prev_eq, num_cmp, num_digits).as(boolean_t);
}

NdArrayRef CompareProtocol::TraversalAND(const NdArrayRef& cmp, NdArrayRef eq,
                                         size_t num_input, size_t num_digits) {
  if (absl::has_single_bit(num_digits)) {
    // If the num_digits is a power of two,
    // call the TraversalAND and return
    return TraversalANDFullBinaryTree(cmp, eq, num_input, num_digits);
  }

  // If the num_digits is not a power of two, we split the input into many sub-
  // full binary trees. And traversal on the subtrees consequently
  size_t remain_num_digits = num_digits;
  size_t current_num_digits = absl::bit_floor(remain_num_digits);

  // Save the current_num_digits data for each input
  NdArrayRef current_cmp(
      cmp.eltype(), {static_cast<int64_t>(current_num_digits * num_input)});
  NdArrayRef current_eq(eq.eltype(), current_cmp.shape());
  pforeach(0, num_input, [&](int64_t idx) {
    std::memcpy(&current_cmp.at(idx * current_num_digits),
                &cmp.at(idx * num_digits), current_num_digits * cmp.elsize());
    std::memcpy(&current_eq.at(idx * current_num_digits),
                &eq.at(idx * num_digits), current_num_digits * eq.elsize());
  });

  auto cmp_res =
      TraversalAND(current_cmp, current_eq, num_input, current_num_digits);

  // Update remain num digits
  remain_num_digits = remain_num_digits - current_num_digits + 1;

  while (remain_num_digits != 1) {
    current_num_digits = absl::bit_floor(remain_num_digits);
    current_cmp = NdArrayRef(
        cmp.eltype(), {static_cast<int64_t>(current_num_digits * num_input)});
    current_eq = NdArrayRef(eq.eltype(), current_cmp.shape());

    pforeach(0, num_input, [&](int64_t idx) {
      // Copy the cmp_res to the first digit in current_cmp
      std::memcpy(&current_cmp.at(idx * current_num_digits), &cmp_res.at(idx),
                  cmp.elsize());

      // Copy the next (current_num_digits - 1) digits from cmp array
      // which has not been compared
      std::memcpy(&current_cmp.at(idx * current_num_digits + 1),
                  &cmp.at((idx + 1) * num_digits - remain_num_digits + 1),
                  (current_num_digits - 1) * cmp.elsize());

      // We skip the left-most eq_res, which is unnecessary for next loop

      // Copy the next (current_num_digits - 1) digits from eq array
      // which has not been compared
      std::memcpy(&current_eq.at(idx * current_num_digits + 1),
                  &eq.at((idx + 1) * num_digits - remain_num_digits + 1),
                  (current_num_digits - 1) * eq.elsize());
    });

    cmp_res =
        TraversalAND(current_cmp, current_eq, num_input, current_num_digits);
    remain_num_digits = remain_num_digits - current_num_digits + 1;
  }
  return cmp_res;
}

std::array<NdArrayRef, 2> CompareProtocol::TraversalANDWithEq(
    const NdArrayRef& cmp, const NdArrayRef& eq, size_t num_input,
    size_t num_digits) {
  if (absl::has_single_bit(num_digits) == 1) {
    return TraversalANDWithEqFullBinaryTree(cmp, eq, num_input, num_digits);
  }

  // NOTE(jingyu):
  //  If the num_digits is not a power of two,
  //  1.Init the remain_num_digits = num_digits;
  //  2.We choose the the maximal value current_num_digits,
  //    such that current_num_digits is a power of two
  //    and current_num_digits <= remain_num_digitis.
  //  3.Call TraversalANDWithEq function to compare the current_num_digits.
  //    (remain_num_digits -= current_num_digits);
  //  4.Add the one digit compare result to remain_num_digits;
  //  5.Loop 2 - 4 until the remain_num_digits = 1, get the final result;
  //  We need to use this method in num_input
  // For example num_digits = 5
  // Input
  // cmp[1]||cmp[2]||cmp[3]||cmp[4]||cmp[5]...
  // eq[1] ||eq[2] ||eq[3] ||eq[4] ||eq[5]...
  // First call TraversalANDWithEq: on [1...4]
  // Add the compare result to the remain digits: 5 - 4 + 1 = 2;
  // cmp_res[1..4]||cmp[5]...
  // eq_res[1..4] ||eq[5]...
  // Second call TraversalANDWithEq: on the remains 2 digits
  // Get the final result

  // Init the remain_num_digits = num_digits
  size_t remain_num_digits = num_digits;

  // Choose the maximal value current_num_digits
  // such that current_num_digits is a power of two
  // and current_num_digits <= remain_num_digitis.
  size_t current_num_digits = absl::bit_floor(remain_num_digits);

  // Save the current_num_digits data for each input
  NdArrayRef current_cmp(
      cmp.eltype(), {static_cast<int64_t>(current_num_digits * num_input)});
  NdArrayRef current_eq(eq.eltype(), current_cmp.shape());

  // Copy from cmp to current_cmp
  // Copy from eq to current_eq
  pforeach(0, num_input, [&](int64_t idx) {
    std::memcpy(&current_cmp.at(idx * current_num_digits),
                &cmp.at(idx * num_digits), current_num_digits * cmp.elsize());
    std::memcpy(&current_eq.at(idx * current_num_digits),
                &eq.at(idx * num_digits), current_num_digits * eq.elsize());
  });

  // Calculate the low current_num_digits in cmp result
  NdArrayRef cmp_res;
  NdArrayRef eq_res;
  auto [_cmp_res, _eq_res] = TraversalANDWithEq(current_cmp, current_eq,
                                                num_input, current_num_digits);
  // NOTE(lwj): auto unbox is a C++20 feature
  cmp_res = _cmp_res;
  eq_res = _eq_res;

  // Update the remain_num_digits
  remain_num_digits = remain_num_digits - current_num_digits + 1;

  while (remain_num_digits != 1) {
    current_num_digits = absl::bit_floor(remain_num_digits);
    current_cmp = NdArrayRef(
        cmp.eltype(), {static_cast<int64_t>(current_num_digits * num_input)});
    current_eq = NdArrayRef(eq.eltype(), current_cmp.shape());

    pforeach(0, num_input, [&](int64_t idx) {
      // Copy the cmp_res to the first digit in current_cmp
      std::memcpy(&current_cmp.at(idx * current_num_digits), &cmp_res.at(idx),
                  cmp.elsize());

      // Copy the next (current_num_digits - 1) digits from cmp array
      // which has not been compared
      std::memcpy(&current_cmp.at(idx * current_num_digits + 1),
                  &cmp.at((idx + 1) * num_digits - remain_num_digits + 1),
                  (current_num_digits - 1) * cmp.elsize());

      // Copy from eq_res
      std::memcpy(&current_eq.at(idx * current_num_digits), &eq_res.at(idx),
                  eq.elsize());

      // Copy from eq
      std::memcpy(&current_eq.at(idx * current_num_digits + 1),
                  &eq.at((idx + 1) * num_digits - remain_num_digits + 1),
                  (current_num_digits - 1) * eq.elsize());
    });
    auto [_cmp, _eq] = TraversalANDWithEq(current_cmp, current_eq, num_input,
                                          current_num_digits);
    cmp_res = _cmp;
    eq_res = _eq;
    remain_num_digits = remain_num_digits - current_num_digits + 1;
  }
  return {cmp_res, eq_res};
}

std::array<NdArrayRef, 2> CompareProtocol::TraversalANDWithEqFullBinaryTree(
    NdArrayRef cmp, NdArrayRef eq, size_t num_input, size_t num_digits) {
  SPU_ENFORCE(cmp.shape().size() == 1, "need 1D array");
  SPU_ENFORCE_EQ(cmp.shape(), eq.shape());
  SPU_ENFORCE_EQ(cmp.numel(), eq.numel());
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)cmp.numel());

  for (size_t i = 1; i <= num_digits; i += 1) {
    int64_t current_num_digits = num_digits / (1 << (i - 1));
    if (current_num_digits == 1) {
      break;
    }
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    int64_t n = current_num_digits * num_input;
    auto lhs_eq = eq.slice({0}, {n}, {2});
    auto rhs_eq = eq.slice({1}, {n}, {2});

    auto lhs_cmp = cmp.slice({0}, {n}, {2});
    auto rhs_cmp = cmp.slice({1}, {n}, {2});

    // Correlated ANDs
    //   _eq = rhs_eq & lhs_eq
    //  _cmp = rhs_eq & lhs_cmp
    auto [_eq, _cmp] =
        basic_ot_prot_->CorrelatedBitwiseAnd(rhs_eq, lhs_eq, lhs_cmp);
    eq = _eq;
    cmp = ring_xor(_cmp, rhs_cmp);
  }

  return {cmp, eq};
}

NdArrayRef CompareProtocol::TraversalANDFullBinaryTree(NdArrayRef cmp,
                                                       NdArrayRef eq,
                                                       size_t num_input,
                                                       size_t num_digits) {
  // Tree-based traversal ANDs
  // lt0[0], lt0[1], ..., lt0[M],
  // lt1[0], lt1[1], ..., lt1[M],
  // ...
  // ltN[0], ltN[1], ..., ltN[M],
  //
  // View two slices as Nx(M/2) matrix
  // Each row contains M/2 digits.
  // Slice0 contains the even digits
  // slice0: lt0[0], lt0[2], ..., lt0[2*j]
  //         lt1[0], lt1[2], ..., lt1[2*j]
  //         ....
  //         ltn[0], ltn[2], ..., ltn[2*j]
  //
  // Slice1 contains the odd digits
  // slice1: lt0[1], lt0[3], ..., lt0[2*j+1]
  //         lt1[1], lt0[3], ..., lt0[2*j+1]
  //         ....
  //         ltn[1], ltn[3], ..., ltn[2*j+1]
  SPU_ENFORCE(cmp.shape().size() == 1, "need 1D Array");
  SPU_ENFORCE_EQ(cmp.shape(), eq.shape());
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)cmp.numel());

  for (size_t i = 1; i <= num_digits; i += 1) {
    size_t current_num_digits = num_digits / (1 << (i - 1));
    if (current_num_digits == 1) {
      break;
    }
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    int64_t n = current_num_digits * num_input;

    auto lhs_eq = eq.slice({0}, {n}, {2});
    auto rhs_eq = eq.slice({1}, {n}, {2});
    auto lhs_cmp = cmp.slice({0}, {n}, {2});
    auto rhs_cmp = cmp.slice({1}, {n}, {2});

    if (current_num_digits == 2) {
      cmp = basic_ot_prot_->BitwiseAnd(lhs_cmp, rhs_eq);
      ring_xor_(cmp, rhs_cmp);
      // We skip the ANDs for eq on the last loop
      continue;
    }

    // We skip the AND on the 0-th digit which is unnecessary for the next loop.
    int64_t nrow = num_input;
    int64_t ncol = current_num_digits / 2;
    SPU_ENFORCE_EQ(lhs_eq.numel(), nrow * ncol);

    Shape subshape = {nrow * (ncol - 1)};
    NdArrayRef _lhs_eq(lhs_eq.eltype(), subshape);
    NdArrayRef _rhs_eq(rhs_eq.eltype(), subshape);
    NdArrayRef _lhs_cmp(lhs_cmp.eltype(), subshape);

    NdArrayRef _lhs_cmp_col0(lhs_cmp.eltype(), {static_cast<int64_t>(nrow)});
    NdArrayRef _rhs_eq_col0(rhs_eq.eltype(), _lhs_cmp_col0.shape());

    // Skip the 0-th column and take the remains columns
    // TODO(lwj): Can we have a better way to avoid such copying?
    for (int64_t r = 0; r < nrow; ++r) {
      std::memcpy(&_rhs_eq_col0.at(r), &rhs_eq.at(r * ncol), rhs_eq.elsize());
      std::memcpy(&_lhs_cmp_col0.at(r), &lhs_cmp.at(r * ncol),
                  lhs_cmp.elsize());

      for (int64_t c = 1; c < ncol; ++c) {
        std::memcpy(&_lhs_cmp.at(r * (ncol - 1) + c - 1),
                    &lhs_cmp.at(r * ncol + c), lhs_eq.elsize());
        std::memcpy(&_lhs_eq.at(r * (ncol - 1) + c - 1),
                    &lhs_eq.at(r * ncol + c), lhs_eq.elsize());
        std::memcpy(&_rhs_eq.at(r * (ncol - 1) + c - 1),
                    &rhs_eq.at(r * ncol + c), rhs_eq.elsize());
      }
    }

    // Normal AND on 0-th column
    auto _next_cmp_col0 =
        basic_ot_prot_->BitwiseAnd(_rhs_eq_col0, _lhs_cmp_col0);

    // Correlated AND on the remain columns
    auto [_next_cmp, _next_eq] =
        basic_ot_prot_->CorrelatedBitwiseAnd(_rhs_eq, _lhs_cmp, _lhs_eq);

    // Concat two ANDs
    eq = NdArrayRef(eq.eltype(), {_next_cmp_col0.numel() + _next_cmp.numel()});
    cmp = NdArrayRef(cmp.eltype(), eq.shape());

    for (int64_t r = 0; r < nrow; ++r) {
      std::memcpy(&cmp.at(r * ncol), &_next_cmp_col0.at(r), cmp.elsize());

      for (int64_t c = 1; c < ncol; ++c) {
        std::memcpy(&cmp.at(r * ncol + c),
                    &_next_cmp.at(r * (ncol - 1) + c - 1), _next_cmp.elsize());

        std::memcpy(&eq.at(r * ncol + c), &_next_eq.at(r * (ncol - 1) + c - 1),
                    _next_eq.elsize());
      }
    }

    ring_xor_(cmp, rhs_cmp);
  }

  return cmp;
}

NdArrayRef CompareProtocol::Compute(const NdArrayRef& inp, bool greater_than,
                                    int64_t bitwidth) {
  int64_t bw = SizeOf(inp.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  if (bitwidth == 0) {
    bitwidth = bw;
  }
  // NOTE(lwj): reshape might need copy
  auto flatten = inp.reshape({inp.numel()});
  return DoCompute(flatten, greater_than, nullptr, bitwidth)
      .reshape(inp.shape());
}

NdArrayRef CompareProtocol::BatchCompute(const NdArrayRef& inp,
                                         bool greater_than, int64_t numel,
                                         int64_t bitwidth, int64_t batch_size) {
  int64_t bw = SizeOf(inp.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  if (is_sender_) {
    SPU_ENFORCE_EQ(inp.numel(), numel * batch_size);
  } else {
    SPU_ENFORCE_EQ(inp.numel(), numel);
  }

  if (bitwidth == 0) {
    bitwidth = bw;
  }

  auto flatten = inp.reshape({inp.numel()});
  return DoBatchCompute(flatten, greater_than, numel, bitwidth, batch_size);
}

std::array<NdArrayRef, 2> CompareProtocol::ComputeWithEq(const NdArrayRef& inp,
                                                         bool greater_than,
                                                         int64_t bitwidth) {
  int64_t bw = SizeOf(inp.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  if (bitwidth == 0) {
    bitwidth = bw;
  }
  NdArrayRef eq;
  // NOTE(lwj): reshape might need copy
  auto flatten = inp.reshape({inp.numel()});
  auto cmp = DoCompute(flatten, greater_than, &eq, bitwidth);
  return {cmp.reshape(inp.shape()), eq.reshape(inp.shape())};
}

}  // namespace spu::mpc::bumblebee
