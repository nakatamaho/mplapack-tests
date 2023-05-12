#include <mpfr.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <iostream>
#include <chrono>

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

void matmul_mpfr(long m, long n, long k, mpfr_t alpha, mpfr_t *A, long lda, mpfr_t *B, long ldb, mpfr_t beta, mpfr_t *C, long ldc) {
    mpfr_t sum, temp;
    mpfr_init(sum);
    mpfr_init(temp);

    // C = alpha * A * B + beta * C
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            mpfr_set_ui(sum, 0.0, MPFR_RNDN);
            for (long l = 0; l < k; l++) {
                mpfr_mul(temp, A[i + l * ldb], B[l + j * ldb], MPFR_RNDN);
                mpfr_add(sum, sum, temp, MPFR_RNDN);
            }
            mpfr_mul(sum, sum, alpha, MPFR_RNDN);
            mpfr_mul(C[i + j * ldc], C[i + j * ldc], beta, MPFR_RNDN);
            mpfr_add(C[i + j * ldc], C[i + j * ldc], sum, MPFR_RNDN);
        }
    }
    mpfr_clear(sum);
    mpfr_clear(temp);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s m n k prec\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int prec = atoi(argv[4]);

    mpfr_set_default_prec(prec);

    int lda = m;
    int ldb = k;
    int ldc = m;

    mpfr_t *A = (mpfr_t *)malloc(sizeof(mpfr_t) * m * k);
    mpfr_t *B = (mpfr_t *)malloc(sizeof(mpfr_t) * k * n);
    mpfr_t *C = (mpfr_t *)malloc(sizeof(mpfr_t) * m * n);

    mpfr_t alpha, beta;
    mpfr_init(alpha);
    mpfr_init(beta);

    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, 42);

    for (int i = 0; i < m * k; i++) {
        mpfr_init(A[i]);
        mpfr_urandomb(A[i], state);
    }
    for (int i = 0; i < k * n; i++) {
        mpfr_init(B[i]);
        mpfr_urandomb(B[i], state);
    }
    for (int i = 0; i < m * n; i++) {
        mpfr_init(C[i]);
        mpfr_urandomb(C[i], state);
    }

    mpfr_urandomb(alpha, state);
    mpfr_urandomb(beta, state);

    // Compute C = alpha AB + beta C \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_mpfr((long)m, (long)n, (long)k, alpha, A, (long)lda, B, (long)ldb, beta, C, (long)ldc);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    printf("    m     n     k     MFLOPS  Elapsed(s) \n");
    printf("%5d %5d %5d %10.3f  %5.3f\n", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS, elapsed_seconds.count());

    for (int i = 0; i < m * k; i++) {
        mpfr_clear(A[i]);
    }
    for (int i = 0; i < k * n; i++) {
        mpfr_clear(B[i]);
    }
    for (int i = 0; i < m * n; i++) {
        mpfr_clear(C[i]);
    }

    mpfr_clear(alpha);
    mpfr_clear(beta);

    free(A);
    free(B);
    free(C);

    gmp_randclear(state);

    return 0;
}
