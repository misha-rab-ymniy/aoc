#ifndef cpu_multi_h
#define cpu_multi_h

#include <atomic>
#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#define ull unsigned long long
string directory = "CPU";

template <typename T>
void computePiNilakanthaThread(T &pi, ull iterations, ull start, ull end)
{
    T sum(0.0);
    int s = 1;

    for (ull i = start; i <= end; i += 2)
    {
        sum += s * (T(4.0) / (T(i) * (i + 1) * (i + 2)));
        s = -s;
    }

    pi = sum;
}

template <typename T>
T computePiNilakanthaMultiThread(ull iterations)
{
    int numThreads = std::thread::hardware_concurrency();

    if (numThreads < 8)
    {
        directory = "CPU_NO_HT";
    }

    std::vector<std::thread> threads(numThreads);
    std::vector<T> partialSums(numThreads);
    multi_prec<MPREC> pi(3.0);

    ull iterationsPerThread = iterations / numThreads;

    for (ull i = 0; i < numThreads; ++i)
    {
        ull start = 2 + (i * iterationsPerThread * 2);
        ull end = start + (iterationsPerThread * 2) - 1;

        // printf("i: %lli iterations: %lli start: %lli end: %lli\n", i, iterationsPerThread, start, end);

        threads[i] = std::thread(computePiNilakanthaThread<T>, std::ref(partialSums[i]), iterations, start, end);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
    // ShowPi(partialSums[0]);

    for (int i = 1; i < numThreads; ++i)
    {
        // ShowPi(partialSums[i]);
        partialSums[0] += partialSums[i];
    }

    pi += partialSums[0];

    return pi;
}

void callMultiNilakanthaMethod(ull iterations = 100000000)
{
    auto startTime = std::chrono::steady_clock::now();
    using MultiPrec = multi_prec<MPREC>;

    MultiPrec piApproximation = computePiNilakanthaMultiThread<MultiPrec>(iterations);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;
    std::cout << "Nilakantha Method: ";

    ShowPi(piApproximation);
    SavePi(directory, "MultiNilakanthaMethod", piApproximation, duration, iterations);
}

#endif