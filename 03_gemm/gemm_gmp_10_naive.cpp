#include <iostream>
#include <chrono>
#include <stdio.h>
#include <gmp.h>
#include <assert.h>
#define CHECKWITHRGEMM

#if defined CHECKWITHRGEMM
    #include <gmpxx.h>
#endif

#include <time.h>

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

void matmul_gmp(long m, long n, long k, mpf_t alpha, mpf_t *a, long lda, mpf_t *b, long ldb, mpf_t beta, mpf_t *c, long ldc) {
    mpf_t sum, temp;
    mpf_init(sum);
    mpf_init(temp);

    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            mpf_set_ui(sum, 0);
            for (long l = 0; l < k; ++l) {
                mpf_mul(temp, a[i + l * lda], b[l + j * ldb]);
                mpf_add(sum, sum, temp);
            }
            mpf_mul(sum, sum, alpha);
            mpf_mul(c[i + j * ldc], c[i + j * ldc], beta);
            mpf_add(c[i + j * ldc], c[i + j * ldc], sum);
        }
    }
    mpf_clear(sum);
    mpf_clear(temp);
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

    mpf_t *a = (mpf_t *)malloc(m * k * sizeof(mpf_t));
    mpf_t *b = (mpf_t *)malloc(k * n * sizeof(mpf_t));
    mpf_t *c = (mpf_t *)malloc(m * n * sizeof(mpf_t));
    mpf_t alpha, beta;

    // Initialize and set random values for a, b, c, alpha, and beta
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, 42);

    mpf_init(alpha);
    mpf_urandomb(alpha, state, prec);
    mpf_init(beta);
    mpf_urandomb(beta, state, prec);

    for (int i = 0; i < m * k; i++) {
        mpf_init(a[i]);
        mpf_urandomb(a[i], state, prec);
    }

    for (int i = 0; i < k * n; i++) {
        mpf_init(b[i]);
        mpf_urandomb(b[i], state, prec);
    }

    for (int i = 0; i < m * n; i++) {
        mpf_init(c[i]);
        mpf_urandomb(c[i], state, prec);
    }

#ifdef _PRINT
    ////////////////////////////////////////////////
    gmp_printf("alpha = %10.128Ff\n", alpha);
    gmp_printf("beta = %10.128Ff\n", beta);

    printf("a = \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            gmp_printf(" %10.128Ff\n", a[i + j * lda]);
        }
        printf("\n");
    }
    printf("b = \n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            gmp_printf(" %10.128Ff\n", b[i + j * ldb]);
        }
        printf("\n");
    }
    printf("c = \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gmp_printf(" %10.128Ff\n", c[i + j * ldc]);
        }
        printf("\n");
    }
    ////////////////////////////////////////////////
#endif

    // Compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_gmp((long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c, (long)ldc);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    printf("    m     n     k     MFLOPS  Elapsed(s) \n");
    printf("%5d %5d %5d %10.3f  %5.3f\n", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS, elapsed_seconds.count());

#ifdef _PRINT
    // Print the result
    printf("c = alpha ab + beta c\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            gmp_printf(" %10.128Ff\n", c[i + j * ldc]);
        }
        printf("\n");
    }
#endif

    // Clear memory
    for (int i = 0; i < m * k; i++) {
        mpf_clear(a[i]);
    }

    for (int i = 0; i < k * n; i++) {
        mpf_clear(b[i]);
    }

    for (int i = 0; i < m * n; i++) {
        mpf_clear(c[i]);
    }

    mpf_clear(alpha);
    mpf_clear(beta);
    gmp_randclear(state);

    free(a);
    free(b);
    free(c);

    return 0;
}
