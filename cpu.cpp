#include "cpu_single.h"
#include "cpu_multi.h"
#define ull unsigned long long

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

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    // std::cout << "Single:\n";
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
    callBBPMethod();
    callPi();
    // std::cout << std::setprecision(30);
    // std::cout << "Double pi:         " << double(0.314159265358979323846264338327) << "e1" << std::endl;
    // std::cout << std::setprecision(30);
    // std::cout << "Float pi:          " << float(0.314159265358979323846264338327) << "e1" << std::endl;
}
