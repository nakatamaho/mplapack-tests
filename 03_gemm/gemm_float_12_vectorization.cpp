#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <cstring>

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

// a vector of 256 / 32 = 8 floats
typedef float vec __attribute__((vector_size(32)));

// a helper function that allocates n vectors and initializes them with zeros
vec *alloc(int n) {
    vec *ptr = (vec *)std::aligned_alloc(32, 32 * n);
    memset(ptr, 0, 32 * n);
    return ptr;
}

void matmul_float(int m, int n, int k, float alpha, float *_a, int lda, float *_b, int ldb, float beta, float *c, int ldc) {
    if (m != n || k != n) {
        printf("m!=n, k!=n are not supported\n");
        exit(-1);
    }
    if (lda != n || ldb != n || ldc != n) {
        printf("lda!=n, ldb!=n, ldc!=n are not supported\n");
        exit(-1);
    }
    if (alpha != 1.0f) {
        printf("alpha !=1 is supported\n");
        exit(-1);
    }
    if (beta != 0.0f) {
        printf("beta !=0 is supported\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i * n + j] = 0.0;

    int nB = (n + 7) / 8; // number of 8-element vectors in a row (rounded up)

    vec *a = alloc(n * nB);
    vec *b = alloc(n * nB);

    // move both matrices to the aligned region
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * nB + j / 8][j % 8] = _a[i * n + j];
            b[i * nB + j / 8][j % 8] = _b[j * n + i]; // <- b is still transposed
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec s{}; // initialize the accumulator with zeros

            // vertical summation
            for (int k = 0; k < nB; k++)
                s += a[i * nB + k] * b[j * nB + k];

            // horizontal summation
            for (int k = 0; k < 8; k++)
                c[i * n + j] += s[k];
        }
    }

    std::free(a);
    std::free(b);
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
    float *c_org = new float[m * n];
    float alpha = 1.0f; // random_float(gen);
    float beta = 0.0f;  // random_float(gen);

    for (long i = 0; i < m * k; i++) {
        a[i] = random_float(gen);
    }
    for (long i = 0; i < k * n; i++) {
        b[i] = random_float(gen);
    }
    for (long i = 0; i < m * n; i++) {
        c_org[i] = c[i] = random_float(gen);
    }

    // compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_float(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;

    char transa = 't', transb = 't';
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c_org, (long)ldc);

    float tmp, tmp2;
    tmp = tmp2 = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tmp2 = abs(c_org[i + j * ldc] - c[j + i * ldc]);
            tmp = std::max(tmp2, tmp);
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
