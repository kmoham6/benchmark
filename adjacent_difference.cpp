#include "/work/karame.mp/hpx_June/hpx_buran/performance/chplx_library/include/chplx.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <hpx/local/init.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/adjacent_difference.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "heat.hpp"

bool chplx_fork_join_executor = false;
hpx::execution::experimental::fork_join_executor *exec = nullptr;

std::int64_t ghosts = 1;
double k = 0.400000;
double dt = 1.000000;
double dx = 1.000000;
std::int64_t nt = 100;
std::int64_t nx = 1000000;

template <typename Policy>
inline void update(Policy &&p, chplx::Array<double, chplx::Domain<1>> &d,
                   chplx::Array<double, chplx::Domain<1>> &d2) {
  auto NX = nx + 1;
  chplx::forall(p, chplx::Range{1, NX - 2}, [&](auto i) {
    d2(i) =
        d(i) + (((dt * k) / (dx * dx)) * ((d(1 + i) + d(1 - i)) - (2 * d(i))));
  });
  d2(0) = d2(NX - 2);
  d2(NX - 2) = d2(1);
}

template <typename Policy>
inline void heat_chplx(Policy &&p, auto max_elements) {

  nx = max_elements;
  auto NX = nx - 1;
  chplx::Array<double, chplx::Domain<1>> data(chplx::Range(0, NX));
  chplx::Array<double, chplx::Domain<1>> data2(chplx::Range(0, NX));
  chplx::forall(p, chplx::Range{0, NX}, [&](auto i) {
    data(i) = 1 + (((i - 1) + nx) % nx);
    data2(i) = 0;
  });
  //   hpx::chrono::high_resolution_timer t;
  for (int i = 0; i < nt; ++i) {
    update(p, data, data2);
  };
  //   const auto elapsed = t.elapsed();
  //   std::cout << "chapelng," << nx << "," << nt << ","
  //             << hpx::resource::get_num_threads() << "," << dt << "," << dx
  //             << ","
  //             << elapsed << ",0\n";
}

void measureAdjacent_differenceAlgorithms() {

  // auto  chunk_size = 4;

  std::size_t start = 64;
  std::size_t till = 1 << 25;

  const auto NUM_ITERATIONS = 10;

  std::vector<std::array<double, 4>> data;
  std::ofstream fout(
      "/work/karame.mp/hpx_June/hpx_buran/performance/result_forall.csv",
      std::ios_base::app);
  fout << "s,seq,par,speedUp\n";
  for (size_t s = start; s <= till; s *= 2) {
    // std::vector<std::string> arr = generateStrings (s, "string");
    // std::iota(std::begin(arr), std::end(arr), 1, [](int value) {
    //   return std::to_string (value);
    std::vector<int> arr(s);
    std::iota(std::begin(arr), std::end(arr), 1);

    //  chunk_size *= 2;
    hpx::execution::experimental::static_chunk_size scs;
    hpx::execution::experimental::num_cores nc(64);
    hpx::execution::experimental::adaptive_core_chunk_size acc;
    // hpx::execution::experimental::auto_chunk_size acs;

    double seqTime = 0;
    double parTime = 0;
    double speedUp = 0;

    for (int i = 0; i <= NUM_ITERATIONS + 5; i++) {
      std::vector<int> res(s);
      auto t1 = std::chrono::high_resolution_clock::now();
      // hpx::adjacent_difference(
      //     hpx::execution::seq, arr.begin(), arr.end(), res.begin())
      //     [](auto x, auto y) { return std::sin(x) - std::cos(y); });

      // hpx::adjacent_difference(hpx::execution::seq, arr.begin(),
      // arr.end(),
      //                          res.begin());
      ///////////////////////////////////////////////////////////////////////
      // / CHPLX
      // constexpr int NX = 100000;
      heat_chplx(hpx::execution::seq, s);

      // .                 std::sin(std::pow(x, 3))),
      //                       5) +
      //              std::exp(std::sqrt(
      //                  std::pow(std::sin(std::cos(x)) *
      //                  std::cos(std::sin(y)),
      //                           6) +
      //                  std::pow(std::tan(std::exp(x * y)), 6))) +
      //              std::log(std::abs(std::sin(std::exp(std::pow(x, 2))) *
      //                                    std::cos(std::exp(std::pow(y,
      // 2))) +
      //                                1e-9)) +
      //              std::atan(std::pow(std::sin(std::exp(x + y)), 4)) *
      //                  std::acosh(std::pow(std::cos(std::exp(x * y)), 4))
      // +
      //              std::pow(
      //                  std::hypot(std::log(std::abs(x)),
      //                  std::log(std::abs(y))), 4) *
      //                  std::exp(std::log1p(std::pow(std::tan(x * y),
      //                  2)));
      //     });

      auto end1 = std::chrono::high_resolution_clock::now();

      if (i < 5) {
        continue;
      }
      std::chrono::duration<double> time_span1 =
          std::chrono::duration_cast<std::chrono::duration<double>>(end1 - t1);
      seqTime += time_span1.count();
    }
    for (int i = 0; i <= NUM_ITERATIONS + 5; i++) {
      std::vector<int> res1(s);
      auto t2 = std::chrono::high_resolution_clock::now();

      // hpx::adjacent_difference(hpx::execution::par.with(std::ref(acc))),
      // //     arr.begin(), arr.end(), res1.begin();
      // hpx::adjacent_difference(hpx::execution::par.with(nc, scs),
      // arr.begin(),
      //                          arr.end(), res1.begin());
      // hpx::adjacent_difference(hpx::execution::par.with(std::ref(acc)),
      //                          arr.begin(), arr.end(), res1.begin());
      // hpx::adjacent_difference(hpx::execution::par, arr.begin(),
      // arr.end(),
      //                          res1.begin());

      ///////////////////////////////////////////////////////////////////////
      /// CHPLX
      // constexpr int NX = 100000;
      heat_chplx(hpx::execution::par.with(std::ref(acc)), s);
      heat_chplx(hpx::execution::par.with(nc, scs), s);
      // chplx::forall(
      //     hpx::execution::par.with(std::ref(acc)), chplx::Range{0, s - 1},
      //     [&](auto i) {
      //       int y = i;
      //       int x = i;
      //       auto abc_res =
      //           std::pow(std::sin(std::tan(std::pow(x, 3)) *
      //                             std::cos(std::pow(y, 3))),
      //                    5) +
      //           std::pow(std::cos(std::tan(std::pow(y, 3)) *
      //                             std::sin(std::pow(x, 3))),
      //                    5) +
      //           std::exp(std::sqrt(
      //               std::pow(std::sin(std::cos(x)) * std::cos(std::sin(y)),
      //               6) + std::pow(std::tan(std::exp(x * y)), 6))) +
      //           std::log(std::abs(std::sin(std::exp(std::pow(x, 2))) *
      //                                 std::cos(std::exp(std::pow(y, 2))) +
      //                             1e-9)) +
      //           std::atan(std::pow(std::sin(std::exp(x + y)), 4)) *
      //               std::acosh(std::pow(std::cos(std::exp(x * y)), 4)) +
      //           std::pow(
      //               std::hypot(std::log(std::abs(x)), std::log(std::abs(y))),
      //               4) *
      //               std::exp(std::log1p(std::pow(std::tan(x * y), 2)));

      //       (void)abc_res;
      //     });
      //////////////////////////////////////////////////////////////////////////////

      // hpx::adjacent_difference(
      //     hpx::execution::par.with(std::ref(acc)), arr.begin(), arr.end(),
      //     res1.begin(), [](auto x, auto y) {
      //       return std::pow(std::sin(std::tan(std::pow(x, 3)) *
      //                                std::cos(std::pow(y, 3))),
      //                       5) +
      //              std::pow(std::cos(std::tan(std::pow(y, 3)) *
      //                                std::sin(std::pow(x, 3))),
      //                       5) +
      //              std::exp(std::sqrt(
      //                  std::pow(std::sin(std::cos(x)) *
      //                  std::cos(std::sin(y)),
      //                           6) +
      //                  std::pow(std::tan(std::exp(x * y)), 6))) +
      //              std::log(std::abs(std::sin(std::exp(std::pow(x, 2))) *
      //                                    std::cos(std::exp(std::pow(y,
      // 2))) +
      //                                1e-9)) +
      //              std::atan(std::pow(std::sin(std::exp(x + y)), 4)) *
      //                  std::acosh(std::pow(std::cos(std::exp(x * y)), 4))
      // +
      //              std::pow(
      //                  std::hypot(std::log(std::abs(x)),
      //                  std::log(std::abs(y))), 4) *
      //                  std::exp(std::log1p(std::pow(std::tan(x * y),
      //                  2)));
      //     });

      auto end2 = std::chrono::high_resolution_clock::now();

      if (i < 5) {
        continue;
      }

      std::chrono::duration<double> time_span2 =
          std::chrono::duration_cast<std::chrono::duration<double>>(end2 - t2);

      parTime += time_span2.count();
    }

    seqTime /= NUM_ITERATIONS;
    parTime /= NUM_ITERATIONS;
    speedUp = seqTime / parTime;

    data.push_back(std::array<double, 4>{(double)s, seqTime, parTime, speedUp});
    std::cout << "n : " << s << '\n';
    std::cout << "seq: " << seqTime << '\n';
    std::cout << "par: " << parTime << '\n';
    std::cout << "spddd: " << speedUp << "\n\n";
    fout << s << "," << seqTime << "," << parTime << "," << speedUp << "\n";

    for (auto &d : data) {

      std::cout << d[0] << "," << d[1] << "," << d[2] << "," << d[3] << ","
                << ",\n";
    }
  }
  fout.close();

  //   }
}
int hpx_main(hpx::program_options::variables_map &) {
  measureAdjacent_differenceAlgorithms();

  return hpx::local::finalize();
}
int main(int argc, char *argv[]) {
  std::vector<std::string> cfg;
  cfg.push_back("hpx.os_threads=all");
  hpx::local::init_params init_args;
  init_args.cfg = cfg;

  // Initialize and run HPX.
  HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
                  "HPX main exited with non-zero status");

  return hpx::util::report_errors();
}

//-------------------------------------start

// void measureAdjacent_differenceAlgorithms() {

//   // auto  chunk_size = 4;

//   std::size_t start = 10000;
//   // std::size_t till = 1 << 23;

//   std::vector<int> res(start);
//   std::vector<int> arr(start);
//   std::iota(std::begin(arr), std::end(arr), 1);

//   auto t1 = std::chrono::high_resolution_clock::now();
//   hpx::adjacent_difference(
//       hpx::execution::seq, arr.begin(), arr.end(), res.begin(),
//       [](auto x, auto y) {
//         return std::pow(std::sin(std::tan(std::pow(x, 3)) *
//                                  std::cos(std::pow(y, 3))),
//                         5) +
//                std::pow(std::cos(std::tan(std::pow(y, 3)) *
//                                  std::sin(std::pow(x, 3))),
//                         5) +
//                std::exp(std::sqrt(
//                    std::pow(std::sin(std::cos(x)) * std::cos(std::sin(y)),
//   6)
//                    + std::pow(std::tan(std::exp(x * y)), 6))) +
//                std::log(std::abs(std::sin(std::exp(std::pow(x, 2))) *
//                                      std::cos(std::exp(std::pow(y, 2))) +
//                                  1e-9)) +
//                std::atan(std::pow(std::sin(std::exp(x + y)), 4)) *
//                    std::acosh(std::pow(std::cos(std::exp(x * y)), 4)) +
//                std::pow(
//                    std::hypot(std::log(std::abs(x)),
//   std::log(std::abs(y))),
//                    4) *
//                    std::exp(std::log1p(std::pow(std::tan(x * y), 2)));
//       });

//   /// CHPLX
//   // constexpr int NX = 1000;
//   // chplx::forall(hpx::execution::seq, chplx::Range{0, start - 1}, [&](auto
//   i)
//   // {
//   //   int y = i;
//   //   int x = i;
//   //   auto abc_res =
//   //       std::pow(std::sin(std::tan(std::pow(x, 3)) * std::cos(std::pow(y,
//   //       3))),
//   //                5) +
//   //       std::pow(std::cos(std::tan(std::pow(y, 3)) * std::sin(std::pow(x,
//   //       3))),
//   //                5) +
//   //       std::exp(std::sqrt(
//   //           std::pow(std::sin(std::cos(x)) * std::cos(std::sin(y)), 6) +
//   //           std::pow(std::tan(std::exp(x * y)), 6))) +
//   //       std::log(std::abs(std::sin(std::exp(std::pow(x, 2))) *
//   //                             std::cos(std::exp(std::pow(y, 2))) +
//   //                         1e-9)) +
//   //       std::atan(std::pow(std::sin(std::exp(x + y)), 4)) *
//   //           std::acosh(std::pow(std::cos(std::exp(x * y)), 4)) +
//   //       std::pow(std::hypot(std::log(std::abs(x)), std::log(std::abs(y))),
//   4)
//   //       *
//   //           std::exp(std::log1p(std::pow(std::tan(x * y), 2)));

//   //   (void)abc_res;
//   // });
//   //////////////////////////////////////////////////////////////////////////////
//   // hpx::adjacent_difference(hpx::execution::seq, arr.begin(), arr.end(),
//   //                          res.begin());

//   auto end1 = std::chrono::high_resolution_clock::now();

//   auto duration =
//       std::chrono::duration_cast<std::chrono::nanoseconds>(end1 -
//       t1).count();
//   double time_per_iteration = static_cast<double>(duration) / start;

//   std::cout << "Total time taken: " << duration << " nanoseconds" <<
//   std::endl; std::cout << "Time taken per iteration: " << time_per_iteration
//             << " nanoseconds" << std::endl;

//   //   }
// }
// int hpx_main(hpx::program_options::variables_map &) {
//   measureAdjacent_differenceAlgorithms();

//   return hpx::local::finalize();
// }
// int main(int argc, char *argv[]) {
//   std::vector<std::string> cfg;
//   cfg.push_back("hpx.os_threads=all");
//   hpx::local::init_params init_args;
//   init_args.cfg = cfg;

//   // Initialize and run HPX.
//   HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
//                   "HPX main exited with non-zero status");

//   return hpx::util::report_errors();
// }
