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
