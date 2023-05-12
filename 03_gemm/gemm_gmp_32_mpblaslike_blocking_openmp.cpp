#include <iostream>
#include <chrono>
#include <stdio.h>
#include <gmp.h>
#include <assert.h>
#include <time.h>
#include <gmpxx.h>

#include "Rgemm.hpp"

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

#define BLOCK_X 32
#define BLOCK_Y 32

// Blocked matrix multiplication
void matmul_gmp_block(long i0, long j0, long m, long n, long k, mpf_class alpha, mpf_class *A, long lda, mpf_class *B, long ldb, mpf_class *C, long ldc) {
    for (long i = i0; i < std::min(i0 + BLOCK_X, m); ++i) {
        for (long j = j0; j < std::min(j0 + BLOCK_Y, n); ++j) {
            mpf_class sum = 0;
            for (long l = 0; l < k; l++) {
                sum += A[i + l * lda] * B[l + j * ldb];
            }
            C[i + j * ldc] = alpha * sum + C[i + j * ldc];
        }
    }
}

// Matrix multiplication kernel with blocking and OpenMP parallelization
void matmul_gmp(long m, long n, long k, mpf_class alpha, mpf_class *A, long lda, mpf_class *B, long ldb, mpf_class beta, mpf_class *C, long ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i + j * ldc] = beta * C[i + j * ldc];
        }
    }
    #pragma omp parallel for
    for (long i0 = 0; i0 < m; i0 += BLOCK_X) {
        for (long j0 = 0; j0 < n; j0 += BLOCK_Y) {
            matmul_gmp_block(i0, j0, m, n, k, alpha, A, lda, B, ldb, C, ldc);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <m> <k> <n> <prec>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int prec = atoi(argv[4]);
    mpf_set_default_prec(prec);
    int lda = k, ldb = n, ldc = n;

    // Initialize and set random values for A, B, C, alpha, and beta
    gmp_randclass r(gmp_randinit_default);
    r.seed(42);

    mpf_class *A = new mpf_class[m * k];
    mpf_class *B = new mpf_class[k * n];
    mpf_class *C = new mpf_class[m * n];
    mpf_class *C_org = new mpf_class[m * n];
    mpf_class alpha, beta;

    alpha = r.get_f(prec);
    beta = r.get_f(prec);
    for (int i = 0; i < m * k; i++) {
        A[i] = r.get_f(prec);
    }

    for (int i = 0; i < k * n; i++) {
        B[i] = r.get_f(prec);
    }

    for (int i = 0; i < m * n; i++) {
        C_org[i] = r.get_f(prec);
        C[i] = C_org[i];
    }

    // Compute C = alpha AB + beta C \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_gmp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    char transa = 'n', transb = 'n';
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, A, (long)lda, B, (long)ldb, beta, C_org, (long)ldc);

    mpf_class tmp;
    tmp = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tmp += abs(C_org[i + j * ldc] - C[i + j * ldc]);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    gmp_printf("   %.F3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_org;

    return 0;
}
