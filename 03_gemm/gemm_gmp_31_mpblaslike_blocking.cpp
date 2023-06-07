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

// Matrix multiplication kernel with blocking
void matmul_gmp(long m, long n, long k, mpf_class alpha, mpf_class *a, long lda, mpf_class *b, long ldb, mpf_class beta, mpf_class *c, long ldc, long BLOCK_SIZE) {
    mpf_class temp;
    long i, j, l;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            c[i + j * ldc] = beta * c[i + j * ldc];
        }
    }

    for (j = 0; j < n; j += BLOCK_SIZE) {
        for (l = 0; l < k; l += BLOCK_SIZE) {
            for (i = 0; i < m; i += BLOCK_SIZE) {
////////////////////////////////////////////////////////////////////////////
                for (long jj = j; jj < std::min(j + BLOCK_SIZE, n); ++jj) {
                    for (long ll = l; ll < std::min(l + BLOCK_SIZE, k); ++ll) {
                        temp = alpha * b[ll + jj * ldb];
                        for (long ii = i; ii < std::min(i + BLOCK_SIZE, m); ++ii) {
                            c[ii + jj * ldc] += temp * a[ii + ll * lda];
                        }
                    }
                }
////////////////////////////////////////////////////////////////////////////
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <m> <k> <n> <prec> <blocksize>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int prec = atoi(argv[4]);
    long block_size = atoi(argv[5]);
    mpf_set_default_prec(prec);
    int lda = k, ldb = n, ldc = n;

    // Initialize and set random values for a, b, c, alpha, and beta
    gmp_randclass r(gmp_randinit_default);
    r.seed(42);

    mpf_class *a = new mpf_class[m * k];
    mpf_class *b = new mpf_class[k * n];
    mpf_class *c = new mpf_class[m * n];
    mpf_class *c_org = new mpf_class[m * n];
    mpf_class alpha, beta;

    alpha = r.get_f(prec);
    beta = r.get_f(prec);
    for (int i = 0; i < m * k; i++) {
        a[i] = r.get_f(prec);
    }

    for (int i = 0; i < k * n; i++) {
        b[i] = r.get_f(prec);
    }

    for (int i = 0; i < m * n; i++) {
        c_org[i] = r.get_f(prec);
        c[i] = c_org[i];
    }

    // Compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_gmp(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    char transa = 'n', transb = 'n';
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha, a, (long)lda, b, (long)ldb, beta, c_org, (long)ldc);

    mpf_class tmp;
    tmp = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tmp += abs(c_org[i + j * ldc] - c[i + j * ldc]);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", m, n, k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    gmp_printf("   %.F3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_org;

    return 0;
}
