#ifndef gpu_single_cuh
#define gpu_single_cuh

#include <chrono>
#include "utilities.h"
#define MPREC 2
#define directory "GPU"

__global__ void computePiLeibniz(multi_prec<MPREC> *PI, int iterations)
{
    multi_prec<MPREC> pi(0.0);
    multi_prec<MPREC> sign(1.0);

    for (int i = 0; i < iterations; i++)
    {
        multi_prec<MPREC> term(sign / (2 * i + 1));
        pi += term;
        sign = -sign;
    }

    pi *= 4;
    *PI = pi;
}

__global__ void computePiBBP(int digits, multi_prec<MPREC> *PI)
{
    multi_prec<MPREC> pi = 0.0;

    for (int k = 0; k <= digits; ++k)
    {
        multi_prec<MPREC> numerator = multi_prec<MPREC>(1.0) / pow(16, k);
        multi_prec<MPREC> term1 = numerator * (multi_prec<MPREC>(4.0) / (8 * k + 1));
        multi_prec<MPREC> term2 = numerator * (multi_prec<MPREC>(2.0) / (8 * k + 4));
        multi_prec<MPREC> term3 = numerator * (multi_prec<MPREC>(1.0) / (8 * k + 5));
        multi_prec<MPREC> term4 = numerator * (multi_prec<MPREC>(1.0) / (8 * k + 6));

        pi += term1 - term2 - term3 - term4;
    }

    *PI = pi;
}

__global__ void computePiNilakantha(int iterations, multi_prec<MPREC> *PI)
{
    unsigned long long i;
    int s = 1;
    multi_prec<MPREC> pi = 3;

    for (i = 2; i <= iterations * 2; i += 2)
    {
        pi += s * (multi_prec<MPREC>(4) / (multi_prec<MPREC>(i) * (i + 1) * (i + 2)));
        s = -s;
    }

    *PI = pi;
}

void callLeibnizMethod(ull iterations = 10000000)
{
    multi_prec<MPREC> h_sum = 0.0;
    multi_prec<MPREC> *d_sum;

    auto startTime = std::chrono::steady_clock::now();

    cudaMalloc((void **)&d_sum, sizeof(multi_prec<MPREC>));
    cudaMemcpy(d_sum, &h_sum, sizeof(multi_prec<MPREC>), cudaMemcpyHostToDevice);

    computePiLeibniz<<<1, 1>>>(d_sum, iterations);

    cudaMemcpy(&h_sum, d_sum, sizeof(multi_prec<MPREC>), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "Leibniz Method:    ";
    ShowPi(h_sum);
    SavePi(directory, "LeibnizMethod", h_sum, duration, iterations);
}

void callBBpMethod()
{
    multi_prec<MPREC> h_sum = 0.0;
    multi_prec<MPREC> *d_sum;
    int digits = 21;

    auto startTime = std::chrono::steady_clock::now();

    cudaMalloc((void **)&d_sum, sizeof(multi_prec<MPREC>));
    cudaMemcpy(d_sum, &h_sum, sizeof(multi_prec<MPREC>), cudaMemcpyHostToDevice);

    computePiBBP<<<1, 1>>>(digits, d_sum);

    cudaMemcpy(&h_sum, d_sum, sizeof(multi_prec<MPREC>), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "BBP Method:        ";
    ShowPi(h_sum);
    SavePi(directory, "BBPMethod", h_sum, duration, digits);
}

void callNilakanthaMethod(ull iterations = 10000000)
{
    multi_prec<MPREC> h_sum = 0.0;
    multi_prec<MPREC> *d_sum;

    auto startTime = std::chrono::steady_clock::now();

    cudaMalloc((void **)&d_sum, sizeof(multi_prec<MPREC>));
    cudaMemcpy(d_sum, &h_sum, sizeof(multi_prec<MPREC>), cudaMemcpyHostToDevice);

    computePiNilakantha<<<1, 1>>>(iterations, d_sum);

    cudaMemcpy(&h_sum, d_sum, sizeof(multi_prec<MPREC>), cudaMemcpyDeviceToHost);

    cudaFree(d_sum);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "Nilakantha Method: ";
    ShowPi(h_sum);
    SavePi(directory, "NilakanthaMethod", h_sum, duration, iterations);
}

#endif