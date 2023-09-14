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

void measureAdjacent_differenceAlgorithms() {

  std::size_t s = 8192;
  std::vector<int> arr(s);
  std::iota(std::begin(arr), std::end(arr), 1);

  hpx::execution::experimental::adaptive_core_chunk_size acc;

  double parTime = 0;
  std::vector<int> res1(s);
  auto t = std::chrono::high_resolution_clock::now();

  hpx::adjacent_difference(hpx::execution::par.with(acc), arr.begin(),
                           arr.end(), res1.begin());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - t);

  parTime += time_span.count();
  std::cout << "par: " << parTime << '\n';
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
