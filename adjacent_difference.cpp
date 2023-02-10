#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/compute.hpp>

#include <utility>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>

int res = 0;
struct gen_int_t
{
    std::mt19937 mersenne_engine{42};
    std::uniform_int_distribution<int> dist_int{1, 1024};
    auto operator()()
    {
        return dist_int(mersenne_engine);
    }
};

struct gen_float_t
{
    std::mt19937 mersenne_engine{42};
    std::uniform_real_distribution<float> dist_float{1, 1024};
    auto operator()()
    {
        return dist_float(mersenne_engine);
    }
} gen_float{};

template <typename ExPolicy>
double test(ExPolicy policy, std::size_t n)
{
    using allocator_type = hpx::compute::host::block_allocator<float>;
    using executor_type = hpx::compute::host::block_executor<>;

    auto numa_domains = hpx::compute::host::numa_domains();
    allocator_type alloc(numa_domains);
    executor_type executor(numa_domains);

    hpx::execution::adaptive_core_chunk_size acc;
    hpx::compute::vector<float, allocator_type> nums(n, 0.0, alloc), nums2(n, 0.0, alloc);
    // std::size_t chunk_size = n / 160;
    if constexpr (hpx::is_parallel_execution_policy_v<std::decay_t<ExPolicy>>)
    {
        hpx::generate(hpx::execution::par, nums.begin(), nums.end(), gen_float_t{});

    }
    else
    {
        hpx::generate(hpx::execution::seq, nums.begin(), nums.end(), gen_float_t{});
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (hpx::is_parallel_execution_policy_v<ExPolicy>)
        hpx::adjacent_difference(policy.on(executor).with(acc), nums.begin(), nums.end(),
                                 nums2.begin());
    else
        hpx::adjacent_difference(policy, nums.begin(), nums.end(),
                                 nums2.begin());
    // hpx::adjacent_difference(policy.with(cs), nums.begin(), nums.end(),
    // nums2.begin(), [](auto x, auto y){return std::sin(x) - std::cos(y);});
    auto t2 = std::chrono::high_resolution_clock::now();

    res += hpx::count(hpx::execution::par, nums2.begin(), nums2.end(), gen_float_t{}());

    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

template <typename ExPolicy>
auto test3(ExPolicy policy, std::size_t iterations, std::size_t n)
{
    double avg_time = 0.0;
    for (std::size_t i = 0; i < iterations; i++)
    {
        avg_time += test<ExPolicy>(policy, n);
    }
    avg_time /= (double)iterations;
    return avg_time;
}

int hpx_main()
{
    std::cout << "Hello HPX! \n";
    std::cout << "Threads : " << hpx::get_os_thread_count() << '\n';
    std::ofstream fout("result_.csv");
    fout << "n,seq,par,speed_up\n";
    for (std::size_t i = 10; i <= 22; i++)
    {
        std::size_t n = std::pow(2, i);
        double seq = test3(hpx::execution::seq, 2, n);
        double par = test3(hpx::execution::par, 2, n);
        std::cout << "n : " << i << "\t";
        std::cout << "seq : " << seq << "\t";
        std::cout << "par : " << par << "\t";
        std::cout << "spd : " << seq / par << "\n";
        fout << n << ","
             << seq << ","
             << par << ","
             << seq / par << "\n";
    }
    std::cout << "DUMP : " << res << "\n";
    fout.close();
    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    return hpx::init(argc, argv);
}
