#include <iostream>
#include <cstring>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <time.h>

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

// AVX2 = 256bytes; a vector of 256 / 64 = 4 doubles = 4 * 8 = 32 bytes; 
typedef double vec __attribute__((vector_size(32)));

//alloc doubles in 64 bit alignment
vec *alloc(int n) {
    vec *ptr = (vec *)std::aligned_alloc(64, 32 * n);
    memset(ptr, 0, 32 * n);
    return ptr;
}

void matmul_double(long m, long n, long k, double alpha, double *a, long lda, double *b, long ldb, double beta, double *c, long ldc) {
    long nB = (k + 3) / 4; // number of 4-element vectors in a row (rounded up)

    vec *A = alloc(m * nB);
    vec *B = alloc(k * nB);

    for (long j = 0; j < k; j++) {
        for (long i = 0; i < m; i++) {
            A[i * nB + j / 4][j % 4] = a[i + j * lda];
        }
    }

    for (long j = 0; j < n; j++) {
        for (long i = 0; i < k; i++) {
            B[i * nB + j / 4][j % 4] = b[i + j * ldb];
        }
    }

    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            vec s{}; // initialize the accumulator with zeros

            for (long p = 0; p < nB; p++)
                s += A[i * nB + p] * B[j * nB + p];

            for (long p = 0; p < 4; p++)
                c[i + j * ldc] = alpha * s[p] + beta * c[i + j * ldc];
        }
    }

    std::free(A);
    std::free(B);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <k> <n>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int lda = m, ldb = k, ldc = m;

    // Initialize and set random values for a, b, c, alpha, and beta
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> random_double(0.0, 1.0);

    double *a = new double[m * k];
    double *b = new double[k * n];
    double *c = new double[m * n];
    double *c_org = new double[m * n];
    double alpha = random_double(gen);
    double beta = random_double(gen);

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

    char transa = 'n', transb = 'n';
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c_org, (long)ldc);

    double tmp;
    tmp = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tmp += abs(c_org[i + j * ldc] - c[i + j * ldc]);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    printf("   %5.3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_org;

    return 0;
}
