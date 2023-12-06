#include "gpu_single.cuh"
#include "gpu_multi.cuh"
#include <vector>

std::vector<ull> input_nilakantha_iterations{100000, 150000, 200000, 500000, 1000000, 2500000, 5000000, 10000000, 25000000, 50000000, /*100000000, 250000000, 400000000, 500000000*/};
std::vector<ull> input_iterations{5000, 10000, 500000, 5000000, 50000000};
std::vector<ull> input_multi_iterations{6000000, 10000000, 25000000, 50000000, 100000000, 300000000, 750000000, 1000000000, 2500000000, 5000000000};

int main()
{
    // std::cout << "Multi:\n";
    // for (auto iteration : input_multi_iterations)
    // {
    //     for (int j = 0; j < 5; j++)
    //     {
    //         callMultiNilakanthaMethod(iteration);
    //     }
    // }
    std::cout << "Single:\n";
    for (auto iteration : input_iterations)
    {
        for (int j = 0; j < 5; j++)
        {
            callLeibnizMethod(iteration);
        }
    }
    // for (auto iteration : input_nilakantha_iterations)
    // {
    //     for (int j = 0; j < 5; j++)
    //     {
    //         callNilakanthaMethod(iteration);
    //     }
    // }
    callBBpMethod();
    callPi();
}