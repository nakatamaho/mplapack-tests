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
    char transa = 'n', transb = 'n';
    auto start = std::chrono::high_resolution_clock::now();
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c_org, (long)ldc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    printf("    m     n     k     MFLOPS        Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_org;

    return 0;
}
