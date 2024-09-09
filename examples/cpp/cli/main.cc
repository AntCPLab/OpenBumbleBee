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

#include <iostream>

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "seal/seal.h"
#include "yacl/link/link.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/alg.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/dialect/cheetor/cheetor_mul_prot.h"
#include "libspu/mpc/cheetah/dialect/cheetor/prime_ring_cast_prot.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/ring_ext_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

llvm::cl::opt<int> Rank("rank", llvm::cl::init(0),
                        llvm::cl::desc("Rank. 0 for P0, and 1 for P1"));

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<int> P0Width("b0", llvm::cl::init(32),
                           llvm::cl::desc("Bitwidth for P0 (1 <= b0 <= 128)"));

llvm::cl::opt<int> P1Width("b1", llvm::cl::init(32),
                           llvm::cl::desc("Bitwidth for P1 (1 <= b1 <= 128)"));

llvm::cl::opt<std::string> Protocol(
    "prot", llvm::cl::init("CMP"),
    llvm::cl::desc("Protocol:\n"
                   "All (run all protocols)\n"
                   "OLE2k (oblivious linear evaluation over 2^k)\n"
                   "OLEp (oblivious linear evaluation over p)\n"
                   "NExp (negative exponent)\n"
                   "CMP (comparison)\n"
                   "R2R (ring-to-ring)\n"
                   "R2F (ring-to-field)\n"
                   "F2R (field-to-ring)\n"
                   "TRC2k (truncate over Z2k)\n"));

bool CheckRank(int rank) { return rank == 0 or rank == 1; }

bool CheckWidth(int width) { return width >= 1 and width <= 128; }

spu::FieldType BitwidthToFieldType(int bw) {
  if (bw <= 32) {
    return spu::FM32;
  }
  if (bw <= 64) {
    return spu::FM64;
  }
  return spu::FM128;
}

std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                              size_t rank) {
  yacl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

void RunNExp(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
             int) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  spu::NdArrayRef input = spu::mpc::ring_rand(ft, shape).as(
      spu::makeType<spu::mpc::cheetah::AShrTy>(ft));

  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements with bitwidth={} ...",
      __func__, shape.numel(), bitwidth);
  yacl::ElapsedTimer timer;
  spu::RuntimeConfig conf;
  conf.set_field(ft);
  conf.set_fxp_fraction_bits(18);
  conf.mutable_cheetah_2pc_config()->set_ot_kind(
      spu::CheetahOtKind::YACL_Ferret);
  std::shared_ptr<spu::SPUContext> obj =
      spu::mpc::makeCheetahProtocol(conf, link);
  spu::KernelEvalContext kcontext(obj.get());

  // spu::mpc::cheetah::comp(basic_ot_prot);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  auto output = spu::mpc::cheetor::NExp_8(&kcontext, input, 18);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

void RunCMP(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
            int) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  spu::NdArrayRef input = spu::mpc::ring_rand(ft, shape);

  auto comm = std::make_shared<spu::mpc::Communicator>(link);
  auto basic_ot_prot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
      comm, spu::CheetahOtKind::YACL_Ferret);
  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements with bitwidth={} ...",
      __func__, shape.numel(), bitwidth);
  yacl::ElapsedTimer timer;
  spu::mpc::cheetah::CompareProtocol comp(basic_ot_prot);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  auto output = comp.Compute(input, /*greater*/ true, bitwidth);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

void RunR2R(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
            int dest_width) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  spu::NdArrayRef input = spu::mpc::ring_rand(ft, shape);

  auto comm = std::make_shared<spu::mpc::Communicator>(link);
  auto basic_ot_prot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
      comm, spu::CheetahOtKind::YACL_Ferret);
  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements from {} bits to {} bits "
      "...",
      __func__, shape.numel(), bitwidth, dest_width);

  yacl::ElapsedTimer timer;
  spu::mpc::cheetah::RingExtendProtocol rext(basic_ot_prot);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  spu::mpc::cheetah::RingExtendProtocol::Meta meta;
  meta.use_heuristic = true;
  meta.src_ring = ft;
  meta.dst_ring = BitwidthToFieldType(dest_width);
  meta.src_width = bitwidth;
  meta.dst_width = dest_width;
  auto output = rext.Compute(input, meta);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

int64_t GetPrimeWidth(spu::FieldType ft) {
  if (ft == spu::FM32) {
    return 31;
  } else if (ft == spu::FM64) {
    return 59;
  } else if (ft == spu::FM128) {
    return 59 * 2;
  }
  SPU_THROW("ft={} is invalid", ft);
}

spu::NdArrayRef GetPrime(spu::FieldType ft) {
  spu::NdArrayRef out = spu::mpc::ring_zeros(ft, {1});
  if (ft == spu::FM32) {
    // 31bit prime
    out.at<uint32_t>(0) = 2147352577ULL;
  } else if (ft == spu::FM64) {
    // 59bit prime
    out.at<uint64_t>(0) = 1152921504606584833ULL;
  } else if (ft == spu::FM128) {
    // (59 + 59)bit
    out.at<uint64_t>(0) = 1152921504606584833ULL;
    out.at<uint64_t>(1) = 1152921504606683137ULL;
  } else {
    SPU_THROW("ft={} is invalid", ft);
  }
  return out;
}

spu::NdArrayRef RandomPrime(spu::FieldType ft, spu::Shape shape) {
  auto prime = GetPrime(ft);
  spu::NdArrayRef input = spu::mpc::ring_rand(ft, shape);
  if (ft == spu::FM32) {
    spu::NdArrayView<uint32_t> x(input);
    seal::Modulus p(prime.at<uint32_t>(0));
    for (int64_t i = 0; i < shape.numel(); ++i) {
      x[i] = seal::util::barrett_reduce_64(static_cast<uint64_t>(x[i]), p);
    }
  } else if (ft == spu::FM64) {
    spu::NdArrayView<uint64_t> x(input);
    seal::Modulus p(prime.at<uint64_t>(0));
    for (int64_t i = 0; i < shape.numel(); ++i) {
      x[i] = seal::util::barrett_reduce_64(x[i], p);
    }
  } else {
    spu::NdArrayView<uint64_t> x(input);
    seal::Modulus p0(prime.at<uint64_t>(0));
    seal::Modulus p1(prime.at<uint64_t>(1));
    for (int64_t i = 0; i < shape.numel(); i += 2) {
      x[i] = seal::util::barrett_reduce_64(x[i], p0);
      x[i + 1] = seal::util::barrett_reduce_64(x[i + 1], p1);
    }
  }
  return input;
}

void RunF2R(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
            int dest_width) {
  auto ft = BitwidthToFieldType(bitwidth);
  int64_t prime_width = GetPrimeWidth(ft);
  spu::Shape shape = {1L << 20};
  spu::NdArrayRef input = RandomPrime(ft, shape);

  auto comm = std::make_shared<spu::mpc::Communicator>(link);
  auto basic_ot_prot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
      comm, spu::CheetahOtKind::YACL_Ferret);
  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements from {}-bit field to "
      "{}-bit ring ...",
      __func__, shape.numel(), prime_width, dest_width);

  yacl::ElapsedTimer timer;
  spu::mpc::cheetor::PrimeRingCastProtocol rext(basic_ot_prot);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  spu::mpc::cheetor::PrimeRingCastProtocol::Meta meta;
  meta.prime = GetPrime(ft);
  meta.prime_width = prime_width;
  meta.dst_ring = BitwidthToFieldType(dest_width);
  meta.dst_width = dest_width;
  auto output = rext.Compute(input, meta);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m, Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

void RunOLE2k(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
              int) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  yacl::set_num_threads(1);
  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements with {}-bit ring ...",
      __func__, shape.numel(), bitwidth);
  spu::NdArrayRef input = RandomPrime(ft, shape);

  yacl::ElapsedTimer timer;
  spu::mpc::cheetah::CheetahMul share_mul(link, false);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  auto output = share_mul.MulOLE(input, link->Rank() == 0);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

void RunOLEp(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
             int) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m with n={} elements with {}-bit field ...",
      __func__, shape.numel(),
      spu::mpc::cheetor::CheetorMulProt::PrimeNumBits(ft));

  spu::NdArrayRef x = spu::mpc::ring_rand(ft, shape);
  // spu::NdArrayRef y = spu::mpc::ring_rand(ft, shape);

  yacl::ElapsedTimer timer;
  spu::mpc::cheetor::CheetorMulProt ole(link);
  ole.LazyInit(ft);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;

  // NOTE(lwj): we perform local conversion from 2^k to p
  // auto output = ole.RingShareToPrimeShareMul(x, y);
  auto output = ole.RingShareToPrimeShareSquare(x);

  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

void RunTRC2k(const std::shared_ptr<yacl::link::Context>& link, int bitwidth,
              int tr) {
  auto ft = BitwidthToFieldType(bitwidth);
  spu::Shape shape = {1L << 20};
  spu::NdArrayRef input = spu::mpc::ring_rand(ft, shape);

  auto comm = std::make_shared<spu::mpc::Communicator>(link);
  auto basic_ot_prot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
      comm, spu::CheetahOtKind::YACL_Ferret);
  yacl::set_num_threads(1);

  SPDLOG_INFO(
      "Running \x1B[31m{}\033[0m over ring with n={} elements with bitwidth={} "
      "and "
      "shift={} ...",
      __func__, shape.numel(), bitwidth, tr);
  yacl::ElapsedTimer timer;
  spu::mpc::cheetah::TruncateProtocol trc(basic_ot_prot);
  spu::mpc::cheetah::TruncateProtocol::Meta meta;
  meta.use_heuristic = true;
  meta.shift_bits = static_cast<size_t>(tr);
  size_t sent = link->GetStats()->sent_bytes;
  size_t recv = link->GetStats()->recv_bytes;
  auto output = trc.Compute(input, meta);
  double time = timer.CountMs();
  sent = link->GetStats()->sent_bytes - sent;
  recv = link->GetStats()->recv_bytes - recv;
  SPDLOG_INFO(
      "\x1B[31m{}\033[0m Done. Took {} ms, sent {} bytes, recev {} bytes",
      __func__, time, sent, recv);
}

using BwChecker = std::function<bool(int, int)>;
using Runner = std::function<void(
    const std::shared_ptr<yacl::link::Context>& link, int, int)>;
inline bool SameBw(int a, int b) { return a == b; }
inline bool Larger0(int a, int b) { return a > b; }
inline bool Larger1(int a, int b) { return a <= b; }

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  int rank = Rank.getValue();
  auto prot = Protocol.getValue();
  int b0 = P0Width.getValue();
  int b1 = P1Width.getValue();

  SPU_ENFORCE(CheckRank(rank), "invalid rank={}", rank);
  SPU_ENFORCE(CheckWidth(b0), "invalid b0={}", b0);
  SPU_ENFORCE(CheckWidth(b1), "invalid b1={}", b1);

  std::unordered_map<std::string, std::tuple<BwChecker, Runner>> funcs;
  funcs.insert({"OLE2k", {SameBw, RunOLE2k}});
  funcs.insert({"OLEp", {SameBw, RunOLEp}});
  funcs.insert({"CMP", {SameBw, RunCMP}});
  funcs.insert({"NExp", {SameBw, RunNExp}});
  funcs.insert({"R2R", {Larger1, RunR2R}});
  funcs.insert({"F2R", {Larger1, RunF2R}});
  funcs.insert({"TRC2k", {Larger0, RunTRC2k}});

  if (prot == "All") {
    auto link = MakeLink(Parties.getValue(), rank);
    int b0 = 32;
    int b1 = 64;
    int b2 = 10;
    int b3 = b0;
    for (const auto& kv : funcs) {
      if (std::get<0>(kv.second)(b0, b1)) {
        std::get<1>(kv.second)(link, b0, b1);
      } else if (std::get<0>(kv.second)(b0, b2)) {
        std::get<1>(kv.second)(link, b0, b2);
      } else if (std::get<0>(kv.second)(b0, b3)) {
        std::get<1>(kv.second)(link, b0, b3);
      }
    }

    return 0;
  }

  if (funcs.find(prot) != funcs.end()) {
    SPU_ENFORCE(std::get<0>(funcs[prot])(b0, b1),
                "b0={} b1={} is invalid for prot={}", b0, b1, prot);
  } else {
    SPU_THROW("prot={} is invalid", prot);
    return 0;
  }

  auto link = MakeLink(Parties.getValue(), rank);
  std::get<1>(funcs[prot])(link, b0, b1);
  return 0;
}
