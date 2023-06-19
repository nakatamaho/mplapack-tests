#include <iostream>
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

void matmul_double(long m, long n, long k, double alpha, double *a, long lda, double *_b, long ldb, double beta, double *c, long ldc) {

    double *b = new double[n * n];

    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            b[i + j * ldb] = _b[j + i * ldb];

    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            c[i + j * ldc] = beta * c[i + j * ldc];

    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            for (long k = 0; k < n; k++)
                c[i + j * ldc] += alpha * a[i + k * lda] * b[k + j * ldb];
}

int main(int argc, char *argv[]) {
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
    Rgemm(&transa, &transb, m, n, k, alpha, a, lda, b, ldb, beta, c_org, ldc);

    double tmp, tmp2;
    tmp = tmp2 = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
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
