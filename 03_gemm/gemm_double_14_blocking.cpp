#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "Rgemm_double.hpp"

#define MFLOPS 1e-6

// cf. https://netlib.org/lapack/lawnspdf/lawn41.pdf p.120
double flops_gemm(long k_i, long m_i, long n_i) {
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

// AVX2 = 256 bits / 64 (bits / double) = 4 doubles. 4 doubles = 32 bytes

#define ___VECTOR_SIZE_IN_BYTES___ 32
#define ___VECTOR_ALIGNMENT_IN_BYTES___ 32
typedef double vec __attribute__((vector_size(___VECTOR_SIZE_IN_BYTES___)));

// a helper function that allocates n vectors and initializes them with zeros
vec *alloc(long n) {
    vec *ptr = (vec *)std::aligned_alloc(___VECTOR_ALIGNMENT_IN_BYTES___, ___VECTOR_SIZE_IN_BYTES___ * n);
    memset(ptr, 0, ___VECTOR_SIZE_IN_BYTES___ * n);
    return ptr;
}

#define ___KERNEL_SIZE_X___ 3
#define ___KERNEL_SIZE_Y___ 16

// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
void kernel(double *a, vec *b, vec *c, long x, long y, long l, long r, long n) {
    vec t[6][2]{}; // will be zero-filled and stored in ymm registers

    for (long k = l; k < r; k++) {
        for (long i = 0; i < 6; i++) {
            // broadcast a[x + i][k] longo a register
            vec gamma = vec{} + a[(x + i) * n + k]; // converts to a broadcast
                                                    // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
            for (long j = 0; j < 2; j++)
                t[i][j] += gamma * b[(k * n + y) / 4 + j]; // converts to an fma
        }
    }
    // write the results back to C
    for (long i = 0; i < 6; i++)
        for (long j = 0; j < 2; j++)
            c[((x + i) * n + y) / 4 + j] += t[i][j];
}

void matmul_double(long m, long n, long k, double alpha, double *_a, long lda, double *_b, long ldb, double beta, double *_c, long ldc) {
    if (m != n || k != n) {
        printf("m!=n, k!=n are not supported\n");
        exit(-1);
    }
    if (lda != n || ldb != n || ldc != n) {
        printf("lda!=n, ldb!=n, ldc!=n are not supported\n");
        exit(-1);
    }

    long nx = (n + 5) / 6 * 6;
    long ny = (n + 7) / 8 * 8;

    double *a = (double *)alloc(nx * ny);
    double *b = (double *)alloc(nx * ny);
    double *c = (double *)alloc(nx * ny);

    for (long i = 0; i < n; i++) {
        memcpy(&a[i * ny], &_a[i * n], 8 * n);
        memcpy(&b[i * ny], &_b[i * n], 8 * n); // we don't need to transpose b this time
    }

    const long s3 = 64;  // how many columns of B to select
    const long s2 = 120; // how many rows of A to select
    const long s1 = 240; // how many rows of B to select

    for (long i3 = 0; i3 < ny; i3 += s3)
        // now we are working with b[:][i3:i3+s3]
        for (long i2 = 0; i2 < nx; i2 += s2)
            // now we are working with a[i2:i2+s2][:]
            for (long i1 = 0; i1 < ny; i1 += s1)
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
                for (long x = i2; x < std::min(i2 + s2, nx); x += 6)
                    for (long y = i3; y < std::min(i3 + s3, ny); y += 8)
                        kernel(a, (vec *)b, (vec *)c, x, y, i1, std::min(i1 + s1, n), ny);

    for (long i = 0; i < n; i++)
        memcpy(&_c[i * n], &c[i * ny], 8 * n);

    std::free(a);
    std::free(b);
    std::free(c);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <k> <n>\n", argv[0]);
        return 1;
    }

    long m = atoi(argv[1]);
    long k = atoi(argv[2]);
    long n = atoi(argv[3]);
    long lda = m, ldb = k, ldc = m;

    // Initialize and set random values for a, b, c, alpha, and beta
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> random_double(0.0, 1.0);

    double *a = new double[m * k];
    double *b = new double[k * n];
    double *c = new double[m * n];
    double *c_org = new double[m * n];
    double alpha = 1.0; // random_double(gen);
    double beta = 0.0;  // random_double(gen);

    for (long i = 0; i < m * k; i++) {
        a[i] = random_double(gen);
    }
    for (long i = 0; i < k * n; i++) {
        b[i] = random_double(gen);
    }
    for (long i = 0; i < m * n; i++) {
        c_org[i] = c[i] = random_double(gen);
    }

    // compute c = alpha ab + beta c \n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_double(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    char transa = 't', transb = 't';
    Rgemm(&transa, &transb, m, n, k, alpha, a, lda, b, ldb, beta, c_org, ldc);

    double tmp, tmp2;
    tmp = tmp2 = 0.0;
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            tmp2 = abs(c_org[i + j * ldc] - c[j + i * ldc]);
            tmp = std::max(tmp2, tmp);
        }
    }

    printf("    m     n     k     MFLOPS      DIFF     Elapsed(s)\n");
    printf("%5d %5d %5d %10.3f", (int)m, (int)n, (int)k, flops_gemm(k, m, n) / elapsed_seconds.count() * MFLOPS);
    printf("   %5.3e", tmp);
    printf("     %5.3f\n", elapsed_seconds.count());

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_org;

    return 0;
}
