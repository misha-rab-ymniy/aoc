#ifndef gpu_multi_cuh
#define gpu_multi_cuh

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#define ull unsigned long long
#define directory "GPU"

__global__ void computePiNilakanthaKernel(multi_prec<MPREC> *partialSums, ull iterations)
{
    ull tid = blockIdx.x * blockDim.x + threadIdx.x;
    ull start = 2 + (tid * iterations * 2);
    ull end = start + ((iterations * 2) - 1);
    //printf("tid: %lli iterations: %lli start: %lli end: %lli\n", tid, iterations, start, end);
    multi_prec<MPREC> sum(0.0);
    int s = 1;

    for (ull i = start; i <= end; i += 2)
    {
        sum += s * (multi_prec<MPREC>(4.0) / (multi_prec<MPREC>(i) * (i + 1) * (i + 2)));
        s = -s;
    }
    partialSums[tid] = sum;
}

multi_prec<MPREC> computePiNilakanthaCUDA(ull iterations)
{
    const int BLOCKS = 50;
    const int THREADS = 64;
    ull iterationsPerThread = iterations / (THREADS * BLOCKS);
    //std::cout << iterationsPerThread << std::endl;

    multi_prec<MPREC> *host = new multi_prec<MPREC>[THREADS * BLOCKS];

    multi_prec<MPREC> *dev;
    cudaMalloc((void **)&dev, THREADS * BLOCKS * sizeof(multi_prec<MPREC>));

    computePiNilakanthaKernel<<<BLOCKS, THREADS>>>(dev, iterationsPerThread);

    cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(multi_prec<MPREC>), cudaMemcpyDeviceToHost);

    multi_prec<MPREC> piApproximation = 3.0;
    for (ull i = 0; i < BLOCKS * THREADS; ++i)
    {
        // ShowPi(host[i]);
        piApproximation += host[i];
    }
    delete[] host;
    cudaFree(dev);

    return piApproximation;
}

void callMultiNilakanthaMethod(ull iterations = 100000000)
{

    auto startTime = std::chrono::steady_clock::now();

    multi_prec<MPREC> piApproximation = computePiNilakanthaCUDA(iterations);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "Nilakantha Method: ";
    ShowPi(piApproximation);
    SavePi(directory, "MultiNilakanthaMethod", piApproximation, duration, iterations);
}

#endif

// Pi:                0.314159265358979323846264338327e1
// Nilakantha Method: 0.3141592653589793238462643383029502884947169398357372738885e1
// Nilakantha Method: 0.31415926535897932384626433830295028849471693982054741748811e1