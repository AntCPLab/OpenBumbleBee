#include <_types/_uint64_t.h>

#include <algorithm>
#include <random>

#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "yacl/utils/elapsed_timer.h"

// 50bits
// 64-1125899904679937-1024
//
void bench(size_t logN, uint64_t prime) {
  seal::util::NTTTables ntt_tbl(logN, prime);
  std::vector<uint64_t> poly(1 << logN);
  std::uniform_int_distribution<uint64_t> uniform(0, prime);
  std::default_random_engine rdv(std::time(0));
  std::generate_n(poly.data(), poly.size(), [&]() { return uniform(rdv); });

  yacl::ElapsedTimer timer;
  for (size_t i = 0; i < 1000000; ++i) {
    seal::util::ntt_negacyclic_harvey_lazy(poly.data(), ntt_tbl);
  }
  printf("%zd-%llu %fus\n", 1UL << logN, prime,
         timer.CountMs() / 1000000 * 1000.);
}

int main() {
  for (size_t logN : {10, 14}) {
    for (uint64_t p : {1125899904679937ULL, 2251799813554177ULL,
                       72057594037338113ULL, 1152921504606584833ULL}) {
      bench(logN, p);
    }
  }
}

// fwd-64-1125899904679937-1024
//                         time:   [4.2551 µs 4.2646 µs 4.2741 µs]
//
// fwd-64-2251799813554177-1024
//                         time:   [4.2500 µs 4.2585 µs 4.2696 µs]
//                         change: [-0.1371% +0.1164% +0.3607%] (p = 0.37 >
//                         0.05) No change in performance detected.
// Found 12 outliers among 100 measurements (12.00%)
//   1 (1.00%) low mild
//   5 (5.00%) high mild
//   6 (6.00%) high severe
//
// inv-64-2251799813554177-1024
//                         time:   [4.3487 µs 4.3566 µs 4.3650 µs]
//                         change: [-0.2547% +0.0722% +0.4149%] (p = 0.68 >
//                         0.05) No change in performance detected.
// Found 13 outliers among 100 measurements (13.00%)
//   8 (8.00%) high mild
//   5 (5.00%) high severe
//
// fwd-64-72057594037338113-1024
//                         time:   [4.2546 µs 4.2631 µs 4.2734 µs]
// Found 6 outliers among 100 measurements (6.00%)
//   5 (5.00%) high mild
//   1 (1.00%) high severe
//
// inv-64-72057594037338113-1024
//                         time:   [4.3415 µs 4.3490 µs 4.3576 µs]
// Found 7 outliers among 100 measurements (7.00%)
//   4 (4.00%) high mild
//   3 (3.00%) high severe
//
// fwd-64-1152921504606584833-1024
//                         time:   [4.2460 µs 4.2529 µs 4.2622 µs]
// Found 7 outliers among 100 measurements (7.00%)
//   3 (3.00%) high mild
//   4 (4.00%) high severe
//
// inv-64-1152921504606584833-1024
//                         time:   [4.3418 µs 4.3518 µs 4.3635 µs]
// Found 11 outliers among 100 measurements (11.00%)
//   5 (5.00%) high mild
//   6 (6.00%) high severe
//
// fwd-64-1125899904679937-16384
//                         time:   [92.363 µs 92.640 µs 92.947 µs]
//                         change: [-1.2521% -0.5681% +0.0348%] (p = 0.09 >
//                         0.05) No change in performance detected.
// Found 12 outliers among 100 measurements (12.00%)
//   9 (9.00%) high mild
//   3 (3.00%) high severe
//
// inv-64-1125899904679937-16384
//                         time:   [97.282 µs 97.431 µs 97.636 µs]
//                         change: [-2.5230% -1.6615% -0.9206%] (p = 0.00 <
//                         0.05) Change within noise threshold.
// Found 11 outliers among 100 measurements (11.00%)
//   5 (5.00%) high mild
//   6 (6.00%) high severe
//
// fwd-64-2251799813554177-16384
//                         time:   [92.925 µs 93.113 µs 93.336 µs]
//                         change: [-1.7847% -1.0775% -0.4380%] (p = 0.00 <
//                         0.05) Change within noise threshold.
// Found 8 outliers among 100 measurements (8.00%)
//   5 (5.00%) high mild
//   3 (3.00%) high severe
//
// inv-64-2251799813554177-16384
//                         time:   [97.787 µs 98.059 µs 98.367 µs]
//                         change: [-0.4999% -0.0283% +0.4079%] (p = 0.91 >
//                         0.05) No change in performance detected.
// Found 6 outliers among 100 measurements (6.00%)
//   1 (1.00%) high mild
//   5 (5.00%) high severe
//
// fwd-64-72057594037338113-16384
//                         time:   [93.092 µs 93.345 µs 93.617 µs]
// Found 3 outliers among 100 measurements (3.00%)
//   3 (3.00%) high mild
//
// inv-64-72057594037338113-16384
//                         time:   [98.106 µs 98.372 µs 98.656 µs]
// Found 5 outliers among 100 measurements (5.00%)
//   3 (3.00%) high mild
//   2 (2.00%) high severe
//
// fwd-64-1152921504606584833-16384
//                         time:   [93.119 µs 93.345 µs 93.591 µs]
// Found 1 outliers among 100 measurements (1.00%)
//   1 (1.00%) high mild
//
// inv-64-1152921504606584833-16384
//                         time:   [97.742 µs 97.963 µs 98.215 µs]
// Found 11 outliers among 100 measurements (11.00%)
