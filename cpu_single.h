#ifndef cpu_single_h
#define cpu_single_h

#include <chrono>
#include <thread>
#include "utilities.h"
#define MPREC 2
#define ull unsigned long long

string isHT()
{
    int numThreads = std::thread::hardware_concurrency();

    if (numThreads < 8)
    {
        return "CPU_NO_HT";
    }

    return "CPU";
}

multi_prec<MPREC> computePiLeibniz(ull iterations)
{
    multi_prec<MPREC> pi(0.0);
    multi_prec<MPREC> sign(1.0);

    for (ull i = 0; i < iterations; i++)
    {
        multi_prec<MPREC> term(sign / (2 * i + 1));
        pi += term;
        sign = -sign;
    }

    pi *= 4;
    return pi;
}

multi_prec<MPREC> computePiBBP(int digits)
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

    return pi;
}

multi_prec<MPREC> computePiNilakantha(unsigned long long iterations)
{
    unsigned long long i;
    int s = 1;
    multi_prec<MPREC> pi = 3;

    for (i = 2; i <= iterations * 2; i += 2)
    {
        pi += s * (multi_prec<MPREC>(4) / (multi_prec<MPREC>(i) * (i + 1) * (i + 2)));
        s = -s;
    }

    return pi;
}

void callLeibnizMethod(ull iteration = 10000000)
{
    ull iterations = iteration;

    auto startTime = std::chrono::steady_clock::now();

    multi_prec<MPREC> piApproximation = computePiLeibniz(iterations);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "Leibniz Method:    ";
    ShowPi(piApproximation);
    SavePi(isHT(), "LeibnizMethod", piApproximation, duration, iterations);
}

void callBBPMethod()
{
    ull iterations = 21;

    auto startTime = std::chrono::steady_clock::now();

    multi_prec<MPREC> piApproximation = computePiBBP(iterations);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "BBP Method:        ";
    ShowPi(piApproximation);
    SavePi(isHT(), "BBPMethod", piApproximation, duration, iterations);
}

void callNilakanthaMethod(unsigned long long iteration = 1000000)
{
    unsigned long long iterations = iteration;

    auto startTime = std::chrono::steady_clock::now();

    multi_prec<MPREC> piApproximation = computePiNilakantha(iterations);

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time: " << duration / 1000 << std::endl;

    std::cout << "Nilakantha Method: ";
    ShowPi(piApproximation);
    SavePi(isHT(), "NilakanthaMethod", piApproximation, duration, iterations);
}

#endif