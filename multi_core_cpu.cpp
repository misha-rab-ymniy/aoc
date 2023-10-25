#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <mutex>

// Функция, которая будет выполняться параллельно
void CalculateSum(double start, double end, double &sum, std::mutex &mutex)
{
    double localSum = 0;
    for (double i = start; i <= end; i++)
    {
        localSum += std::sqrt(i);
    }

    // Захватываем мьютекс для безопасного доступа к переменной sum
    std::lock_guard<std::mutex> lock(mutex);
    sum += localSum;
}

int main()
{
    auto startTime = std::chrono::steady_clock::now();

    constexpr double N = 4000000000.0;
    unsigned int numThreads = std::thread::hardware_concurrency();

    double sum = 0.0;
    std::vector<std::thread> threads;
    std::mutex mutex;

    double range = N / numThreads;
    double start = 1.0;
    double end = range;

    for (unsigned int i = 0; i < numThreads - 1; i++)
    {
        threads.emplace_back(CalculateSum, start, end, std::ref(sum), std::ref(mutex));
        start = end + 1;
        end += range;
    }

    // Последний поток обрабатывает оставшийся диапазон итераций
    threads.emplace_back(CalculateSum, start, N, std::ref(sum), std::ref(mutex));

    // Ожидаем завершения выполнения всех потоков
    for (auto &thread : threads)
    {
        thread.join();
    }

    auto endTime = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Продолжительность работы на всех потоках: " << duration / 1000 << std::endl;

    return 0;
}