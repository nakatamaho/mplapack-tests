#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include "Rgemm_float.hpp"

#define MFLOPS 1e-6

// cf. https://netlib.org/lapack/lawnspdf/lawn41.pdf p.120
float flops_gemm(int k_i, int m_i, int n_i) {
    float adds, muls, flops;
    float k, m, n;
    m = (float)m_i;
    n = (float)n_i;
    k = (float)k_i;
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
    std::uniform_real_distribution<> random_float(0.0, 1.0);

    float *a = new float[m * k];
    float *b = new float[k * n];
    float *c = new float[m * n];
    float alpha = 1.0f;
    float beta = 0.0;

    for (long i = 0; i < m * k; i++) {
        a[i] = random_float(gen);
    }
    for (long i = 0; i < k * n; i++) {
        b[i] = random_float(gen);
    }
    for (long i = 0; i < m * n; i++) {
        c[i] = random_float(gen);
    }

    // Compute C = alpha AB + beta C \n");
    char transa = 'n', transb = 'n';
    auto start = std::chrono::high_resolution_clock::now();
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c, (long)ldc);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    printf("    m     n     k     MFLOPS  Elapsed(s) \n");
    printf("%5d %5d %5d %10.3f  %5.3f\n", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS, elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
