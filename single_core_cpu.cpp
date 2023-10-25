#include <iostream>
#include <sched.h>
#include <cmath>
#include <chrono>

int main()
{

    auto startTime = std::chrono::steady_clock::now();

    // Устанавливаем привязку к 0-му процессору
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    // std::cout << "Код выполняется на процессоре с номером " << sched_getcpu() << std::endl;

    double i = 0;
    double sum = 0;
    while (i < 4000000000)
    {
        i++;
        sum += std::sqrt(i);
    }
    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Продолжительность работы на одном потоке: " << duration / 1000 << std::endl;

    return 0;
}