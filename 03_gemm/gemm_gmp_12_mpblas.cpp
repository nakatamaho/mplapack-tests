#include <iostream>
#include <chrono>
#include <stdio.h>
#include <gmp.h>
#include <assert.h>
#include <time.h>
#include <gmpxx.h>

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

void Mxerbla(const char *srname, int info) {
    fprintf(stderr, " ** On entry to %s parameter number %2d had an illegal value\n", srname, info);
    exit(info);
}
bool Mlsame(const char *a, const char *b) {
    if (toupper(*a) == toupper(*b))
        return true;
    return false;
}

void Rgemm(const char *transa, const char *transb, long const m, long const n, long const k, mpf_class const alpha, mpf_class *a, long const lda, mpf_class *b, long const ldb, mpf_class const beta, mpf_class *c, long const ldc) {
    bool nota = Mlsame(transa, "N");
    bool notb = Mlsame(transb, "N");
    long nrowa = 0;
    if (nota) {
        nrowa = m;
    } else {
        nrowa = k;
    }
    long nrowb = 0;
    if (notb) {
        nrowb = k;
    } else {
        nrowb = n;
    }
    //
    //     Test the input parameters.
    //
    long info = 0;
    if ((!nota) && (!Mlsame(transa, "C")) && (!Mlsame(transa, "T"))) {
        info = 1;
    } else if ((!notb) && (!Mlsame(transb, "C")) && (!Mlsame(transb, "T"))) {
        info = 2;
    } else if (m < 0) {
        info = 3;
    } else if (n < 0) {
        info = 4;
    } else if (k < 0) {
        info = 5;
    } else if (lda < std::max((long)1, nrowa)) {
        info = 8;
    } else if (ldb < std::max((long)1, nrowb)) {
        info = 10;
    } else if (ldc < std::max((long)1, m)) {
        info = 13;
    }
    if (info != 0) {
        Mxerbla("Rgemm ", info);
        return;
    }
    //
    //     Quick return if possible.
    //
    const mpf_class zero = 0.0;
    const mpf_class one = 1.0;
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one))) {
        return;
    }
    //
    //     And if  alpha.eq.zero.
    //
    long j = 0;
    long i = 0;
    if (alpha == zero) {
        if (beta == zero) {
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= m; i = i + 1) {
                    c[(i - 1) + (j - 1) * ldc] = zero;
                }
            }
        } else {
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= m; i = i + 1) {
                    c[(i - 1) + (j - 1) * ldc] = beta * c[(i - 1) + (j - 1) * ldc];
                }
            }
        }
        return;
    }
    //
    //     Start the operations.
    //
    long l = 0;
    mpf_class temp = 0.0;
    if (notb) {
        if (nota) {
            //
            //           Form  C := alpha*A*B + beta*C.
            //
            for (j = 1; j <= n; j = j + 1) {
                if (beta == zero) {
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] = beta * c[(i - 1) + (j - 1) * ldc];
                    }
                }
                for (l = 1; l <= k; l = l + 1) {
                    temp = alpha * b[(l - 1) + (j - 1) * ldb];
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] += temp * a[(i - 1) + (l - 1) * lda];
                    }
                }
            }
        } else {
            //
            //           Form  C := alpha*A**T*B + beta*C
            //
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= m; i = i + 1) {
                    temp = zero;
                    for (l = 1; l <= k; l = l + 1) {
                        temp += a[(l - 1) + (i - 1) * lda] * b[(l - 1) + (j - 1) * ldb];
                    }
                    if (beta == zero) {
                        c[(i - 1) + (j - 1) * ldc] = alpha * temp;
                    } else {
                        c[(i - 1) + (j - 1) * ldc] = alpha * temp + beta * c[(i - 1) + (j - 1) * ldc];
                    }
                }
            }
        }
    } else {
        if (nota) {
            //
            //           Form  C := alpha*A*B**T + beta*C
            //
            for (j = 1; j <= n; j = j + 1) {
                if (beta == zero) {
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] = beta * c[(i - 1) + (j - 1) * ldc];
                    }
                }
                for (l = 1; l <= k; l = l + 1) {
                    temp = alpha * b[(j - 1) + (l - 1) * ldb];
                    for (i = 1; i <= m; i = i + 1) {
                        c[(i - 1) + (j - 1) * ldc] += temp * a[(i - 1) + (l - 1) * lda];
                    }
                }
            }
        } else {
            //
            //           Form  C := alpha*A**T*B**T + beta*C
            //
            for (j = 1; j <= n; j = j + 1) {
                for (i = 1; i <= m; i = i + 1) {
                    temp = zero;
                    for (l = 1; l <= k; l = l + 1) {
                        temp += a[(l - 1) + (i - 1) * lda] * b[(j - 1) + (l - 1) * ldb];
                    }
                    if (beta == zero) {
                        c[(i - 1) + (j - 1) * ldc] = alpha * temp;
                    } else {
                        c[(i - 1) + (j - 1) * ldc] = alpha * temp + beta * c[(i - 1) + (j - 1) * ldc];
                    }
                }
            }
        }
    }
    //
    //     End of Rgemm.
    //
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

    mpf_class *A_mpf_class = new mpf_class[m * k];
    mpf_class *B_mpf_class = new mpf_class[k * n];
    mpf_class *C_mpf_class = new mpf_class[m * n];
    mpf_class alpha_mpf_class = mpf_class(alpha);
    mpf_class beta_mpf_class = mpf_class(beta);

    for (int i = 0; i < m * k; i++) {
        A_mpf_class[i] = mpf_class(A[i]);
    }

    for (int i = 0; i < k * n; i++) {
        B_mpf_class[i] = mpf_class(B[i]);
    }

    for (int i = 0; i < m * n; i++) {
        C_mpf_class[i] = mpf_class(C[i]);
    }

    // Compute C = alpha AB + beta C \n");
    char transa = 'n', transb = 'n';
    auto start = std::chrono::high_resolution_clock::now();
    Rgemm(&transa, &transb, (long)m, (long)n, (long)k, alpha_mpf_class, A_mpf_class, (long)lda, B_mpf_class, (long)ldb, beta_mpf_class, C_mpf_class, (long)ldc);
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
