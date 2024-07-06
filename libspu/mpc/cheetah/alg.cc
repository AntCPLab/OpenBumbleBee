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

#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/trace.h"
#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/conversion.h"
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

    NdArrayRef out = DispatchUnaryFuncWithBatchedInput(
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
