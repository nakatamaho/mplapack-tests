#include <iostream>
#include <chrono>
#include <stdio.h>
#include <gmp.h>
#include <assert.h>

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

void matmul_gmp(long m, long n, long k, mpf_t alpha, mpf_t *a, long lda, mpf_t *b, long ldb, mpf_t beta, mpf_t *c, long ldc, long block_size) {
    mpf_t temp;
    mpf_t sum;
    mpf_init(temp);
    mpf_init(sum);
    long i, j, l;
    long ii, jj, ll;

    mpf_t *block_a = (mpf_t *)malloc(block_size * block_size * sizeof(mpf_t));
    mpf_t *block_b = (mpf_t *)malloc(block_size * block_size * sizeof(mpf_t));

    for (i = 0; i < block_size * block_size; i++) {
        mpf_init(block_a[i]);
        mpf_init(block_b[i]);
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            mpf_mul(c[i + j * ldc], beta, c[i + j * ldc]);
        }
    }

    for (j = 0; j < n; j += block_size) {
        for (l = 0; l < k; l += block_size) {
            // Copy blocks of a and b into block_a and block_b
            for (ii = 0; ii < block_size; ii++) {
                for (jj = 0; jj < block_size; jj++) {
                    if (ii + l < k && jj + j < n) {
                        mpf_set(block_b[ii + jj * block_size], b[(ii + l) + (jj + j) * ldb]);
                    }
                    if (ii + l < m) {
                        mpf_set(block_a[ii + jj * block_size], a[(ii + l) + jj * lda]);
                    }
                }
            }

            for (jj = 0; jj < block_size && jj + j < n; jj++) {
                mpf_mul(temp, alpha, block_b[jj * block_size]);
                for (ii = 0; ii < m; ii++) {
                    mpf_mul(sum, temp, block_a[ii]);
                    mpf_add(c[ii + (jj + j) * ldc], c[ii + (jj + j) * ldc], sum);
                }
            }
        }
    }

    for (i = 0; i < block_size * block_size; i++) {
        mpf_clear(block_a[i]);
        mpf_clear(block_b[i]);
    }

    free(block_a);
    free(block_b);

    mpf_clear(sum);
    mpf_clear(temp);
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <m> <k> <n> <prec> <block_size>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int prec = atoi(argv[4]);
    int block_size = atoi(argv[5]);
    mpf_set_default_prec(prec);
    int lda = m, ldb = k, ldc = m;

    mpf_t *a = (mpf_t *)malloc(m * k * sizeof(mpf_t));
    mpf_t *b = (mpf_t *)malloc(k * n * sizeof(mpf_t));
    mpf_t *c = (mpf_t *)malloc(m * n * sizeof(mpf_t));
    mpf_t alpha, beta;

    mpf_class *_a = new mpf_class[m * k];
    mpf_class *_b = new mpf_class[k * n];
    mpf_class *_c = new mpf_class[m * n];
    mpf_class _alpha, _beta;

    // Initialize and set random values for a, b, c, alpha, and beta
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, 42);

    mpf_init(alpha);
    mpf_urandomb(alpha, state, prec);
    mpf_init(beta);
    mpf_urandomb(beta, state, prec);
    _alpha = mpf_class(alpha);
    _beta = mpf_class(beta);

    for (int i = 0; i < m * k; i++) {
        mpf_init(a[i]);
        mpf_urandomb(a[i], state, prec);
        _a[i] = mpf_class(a[i]);
    }

    for (int i = 0; i < k * n; i++) {
        mpf_init(b[i]);
        mpf_urandomb(b[i], state, prec);
        _b[i] = mpf_class(b[i]);
    }

    for (int i = 0; i < m * n; i++) {
        mpf_init(c[i]);
        mpf_urandomb(c[i], state, prec);
        _c[i] = mpf_class(c[i]);
    }

    // compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_gmp((long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c, (long)ldc, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    char transa = 'n', transb = 'n';
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, _alpha, _a, (long)lda, _b, (long)ldb, _beta, _c, (long)ldc);

    mpf_class tmp;
    tmp = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tmp += abs(mpf_class(c[i + j * ldc]) - _c[i + j * ldc]);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    gmp_printf("   %.F3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    // clear memory
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
    delete[] _a;
    delete[] _b;
    delete[] _c;

    return 0;
}
