#ifndef UTILITIES_H
#define UTILITIES_H
#if defined(__CUDA_ARCH__)
#include "Doubles/src_gpu/multi_prec.h"
#else
#include "Doubles/src_cpu/multi_prec.h"
#endif
#include <unistd.h>
#include <gmp.h>
#include <filesystem>
#define ull unsigned long long
#define MPREC 2
const multi_prec<MPREC> PI = multi_prec<MPREC>("314159265358979323846264338327") / multi_prec<MPREC>("100000000000000000000000000000");

namespace fs = std::filesystem;

template <int T>
void ShowPi(const multi_prec<T> value)
{
    mpf_t mp, mp2;

    mpf_init2(mp, value.getPrec() * 64);
    mpf_init(mp2);

    const double *ptr = value.getData();

    mpf_set_d(mp, ptr[0]);

    for (int x = 1; x < value.getPrec(); x++)
    {
        mpf_set_d(mp2, ptr[x]);
        mpf_add(mp, mp, mp2);
    }

    mpf_out_str(stdout, 10, 32, mp);
    printf("\n");

    mpf_clears(mp, mp2, NULL);
}

template <int T>
void SavePi(string directory, string methodName, const multi_prec<T> value, double time, ull iterations)
{
    FILE *file;
    string fileName = directory + "/" + methodName + ".txt";
    file = fopen(fileName.c_str(), "a");

    if (file == NULL)
    {
        printf("Ошибка открытия файла\n");
        return;
    }
    FILE *original_stdout = stdout;
    stdout = file;
    printf("%lli ", iterations);

    printf("%lf ", time);

    ShowPi(value);


    fflush(stdout);

    stdout = original_stdout;

    fclose(file);
}

void DeleteTxtFiles(const fs::path &directory)
{
    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            fs::remove(entry.path());
            std::cout << "Удален файл: " << entry.path() << std::endl;
        }
        else if (entry.is_directory())
        {
            DeleteTxtFiles(entry.path());
        }
    }
}

void clearTXT()
{
    std::string directory = fs::current_path();

    DeleteTxtFiles(directory);
}

void callPi()
{
    std::cout << "Pi:                ";
    ShowPi(PI);
}

#endif