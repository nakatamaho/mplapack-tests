#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <cpuid.h>

#include "Rgemm_double.hpp"

#define MFLOPS 1e-6

// cf. https://netlib.org/lapack/lawnspdf/lawn41.pdf p.120
double flops_gemm(int k_i, int m_i, int n_i) {
    double adds, muls, flops;
    double k, m, n;
    m = (double)m_i;
    n = (double)n_i;
    k = (double)k_i;
    muls = m * (k + 2) * n;
    adds = m * k * n;
    flops = muls + adds;
    return flops;
}

// AVX2 = 256 bits / 64 (bits / double) = 4 doubles. 4 doubles = 32 bytes
typedef double vec __attribute__((vector_size(32)));

// a helper function that allocates n vectors and initializes them with zeros
vec *alloc(int n) {
    vec *ptr = (vec *)std::aligned_alloc(32, 32 * n);
    memset(ptr, 0, 32 * n);
    return ptr;
}

void matmul_double(long m, long n, long k, double alpha, double *_a, long lda, double *_b, long ldb, double beta, double *c, long ldc) {
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            c[i * n + j] = beta * c[i * n + j];

    long nB = (n + 3) / 4; // number of 4-element vectors in a row (rounded up)

    vec *a = alloc(n * nB);
    vec *b = alloc(n * nB);

    // move both matrices to the aligned region
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            a[i * nB + j / 4][j % 4] = _a[i * n + j];
            b[i * nB + j / 4][j % 4] = _b[j * n + i]; // <- b is still transposed
        }
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            vec s{}; // initialize the accumulator with zeros

            // vertical summation
            for (long k = 0; k < nB; k++)
                s += a[i * nB + k] * b[j * nB + k];

            // horizontal summation
            for (long k = 0; k < 4; k++)
                c[i * n + j] += s[k];
        }
    }

    std::free(a);
    std::free(b);
}

int main(int argc, char *argv[]) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    // Get the value of the cpuid standard level 0x7
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);

    if (ebx & bit_AVX2) {
        printf("AVX2 is supported.\n");
    } else {
        printf("AVX2 is not supported. Exiting...\n");
    }

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <k> <n>\n", argv[0]);
        return 1;
    }

    long m = atoi(argv[1]);
    long k = atoi(argv[2]);
    long n = atoi(argv[3]);
    long lda = m, ldb = k, ldc = m;

    // Initialize and set random values for a, b, c, alpha, and beta
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> random_double(0.0, 1.0);

    double *a = new double[m * k];
    double *b = new double[k * n];
    double *c = new double[m * n];
    double *c_org = new double[m * n];
    double alpha = 1.0f; // random_double(gen);
    double beta = 0.0f;  // random_double(gen);

    for (long i = 0; i < m * k; i++) {
        a[i] = random_double(gen);
    }
    for (long i = 0; i < k * n; i++) {
        b[i] = random_double(gen);
    }
    for (long i = 0; i < m * n; i++) {
        c_org[i] = c[i] = random_double(gen);
    }

    // compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_double(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    char transa = 't', transb = 't';
    Rgemm(&transa, &transb, m, n, k, alpha, a, lda, b, ldb, beta, c_org, ldc);

    double tmp, tmp2;
    tmp = tmp2 = 0.0;
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            tmp2 = abs(c_org[i + j * ldc] - c[j + i * ldc]);
            tmp = std::max(tmp2, tmp);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", (int)m, (int)n, (int)k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    printf("   %5.3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_org;

    return 0;
}
