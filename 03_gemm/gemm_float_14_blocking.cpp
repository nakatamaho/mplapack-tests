#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
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

// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
void kernel(float *a, vec *b, vec *c, int x, int y, int l, int r, int n) {
    vec t[6][2]{}; // will be zero-filled and stored in ymm registers

    for (int k = l; k < r; k++) {
        for (int i = 0; i < 6; i++) {
            // broadcast a[x + i][k] into a register
            vec alpha = vec{} + a[(x + i) * n + k]; // converts to a broadcast
                                                    // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
            for (int j = 0; j < 2; j++)
                t[i][j] += alpha * b[(k * n + y) / 8 + j]; // converts to an fma
        }
    }

    // write the results back to C
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 2; j++)
            c[((x + i) * n + y) / 8 + j] += t[i][j];
}

void matmul_float(int m, int n, int k, float alpha, float *_a, int lda, float *_b, int ldb, float beta, float *_c, int ldc) {
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
    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;

    float *a = (float *)alloc(nx * ny);
    float *b = (float *)alloc(nx * ny);
    float *c = (float *)alloc(nx * ny);

    for (int i = 0; i < n; i++) {
        memcpy(&a[i * ny], &_a[i * n], 4 * n);
        memcpy(&b[i * ny], &_b[i * n], 4 * n); // we don't need to transpose b this time
    }

    const int s3 = 64;  // how many columns of B to select
    const int s2 = 120; // how many rows of A to select
    const int s1 = 240; // how many rows of B to select

    for (int i3 = 0; i3 < ny; i3 += s3)
        // now we are working with b[:][i3:i3+s3]
        for (int i2 = 0; i2 < nx; i2 += s2)
            // now we are working with a[i2:i2+s2][:]
            for (int i1 = 0; i1 < ny; i1 += s1)
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
                for (int x = i2; x < std::min(i2 + s2, nx); x += 6)
                    for (int y = i3; y < std::min(i3 + s3, ny); y += 16)
                        kernel(a, (vec *)b, (vec *)c, x, y, i1, std::min(i1 + s1, n), ny);

    for (int i = 0; i < n; i++)
        memcpy(&_c[i * n], &c[i * ny], 4 * n);

    std::free(a);
    std::free(b);
    std::free(c);
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
