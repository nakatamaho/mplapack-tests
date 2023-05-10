#include <iostream>
#include <chrono>
#include <stdio.h>
#include <gmp.h>
#include <assert.h>
#include <time.h>

gmp_randstate_t state;

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

void matmul_gmp(int m, int n, int k, mpf_t alpha, mpf_t *A, int lda, mpf_t *B, int ldb, mpf_t beta, mpf_t *C, int ldc) {
    mpf_t sum;
    mpf_init(sum);
    mpf_t mul;
    mpf_init(mul);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mpf_set_ui(sum, 0);
            for (int l = 0; l < k; l++) {
                mpf_mul(mul, A[i * lda + l], B[l * ldb + j]);
                mpf_add(sum, sum, mul);
            }
            mpf_mul(sum, alpha, sum);
            mpf_mul(C[i * ldc + j], beta, C[i * ldc + j]);
            mpf_add(C[i * ldc + j], C[i * ldc + j], sum);
            mpf_set_ui(sum, 0);
        }
    }
    mpf_clear(mul);
    mpf_clear(sum);
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
    int lda = k, ldb = n, ldc = n;

    mpf_t *A = (mpf_t *)malloc(m * k * sizeof(mpf_t));
    mpf_t *B = (mpf_t *)malloc(k * n * sizeof(mpf_t));
    mpf_t *C = (mpf_t *)malloc(m * n * sizeof(mpf_t));
    mpf_t alpha, beta;

    // Initialize and set random values for A, B, C, alpha, and beta
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL));

    mpf_init(alpha);
    mpf_urandomb(alpha, state, prec);
    mpf_init(beta);
    mpf_urandomb(beta, state, prec);

    for (int i = 0; i < m * k; i++) {
        mpf_init(A[i]);
        mpf_urandomb(A[i], state, prec);
    }

    for (int i = 0; i < k * n; i++) {
        mpf_init(B[i]);
        mpf_urandomb(B[i], state, prec);
    }

    for (int i = 0; i < m * n; i++) {
        mpf_init(C[i]);
        mpf_urandomb(C[i], state, prec);
    }

    // Compute C = alpha AB + beta C \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_gmp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    printf("    m     n     k     MFLOPS  Elapsed(s) \n");
    printf("%5d %5d %5d %10.3f  %5.3f\n", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS, elapsed_seconds.count());

#ifdef _PRINT_MAT
    // Print the result
    printf("C = alpha AB + beta C\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gmp_printf(" %10.6Ff", C[i * ldc + j]);
        }
        printf("\n");
    }
#endif

    // Clear memory
    for (int i = 0; i < m * k; i++) {
        mpf_clear(A[i]);
    }

    for (int i = 0; i < k * n; i++) {
        mpf_clear(B[i]);
    }

    for (int i = 0; i < m * n; i++) {
        mpf_clear(C[i]);
    }

    mpf_clear(alpha);
    mpf_clear(beta);
    gmp_randclear(state);

    free(A);
    free(B);
    free(C);

    return 0;
}
