#include <iostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <omp.h>
#include <bits/stdc++.h>

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

const int B = 8; // number of elements in a vector
typedef float vector __attribute__((vector_size(4 * B)));

float *alloc(int n) {
    float *ptr = (float *)std::aligned_alloc(64, 4 * n);
    memset(ptr, 0, 4 * n);
    return ptr;
}

// c: 6 x 16
// a: 6 x k
// b: k x 16
// c[x:x+6][y:y+16] += a[x:x+6][l:r] * b[l:r][y:y+16]

void kernel(float *a, vector *b, vector *c, int x, int y, int l, int r, int n) {
    vector t00, t01, t10, t11, t20, t21, t30, t31, t40, t41, t50, t51;

    t00 = c[((x + 0) * n + y) / 8 + 0];
    t01 = c[((x + 0) * n + y) / 8 + 1];

    t10 = c[((x + 1) * n + y) / 8 + 0];
    t11 = c[((x + 1) * n + y) / 8 + 1];

    t20 = c[((x + 2) * n + y) / 8 + 0];
    t21 = c[((x + 2) * n + y) / 8 + 1];

    t30 = c[((x + 3) * n + y) / 8 + 0];
    t31 = c[((x + 3) * n + y) / 8 + 1];

    t40 = c[((x + 4) * n + y) / 8 + 0];
    t41 = c[((x + 4) * n + y) / 8 + 1];

    t50 = c[((x + 5) * n + y) / 8 + 0];
    t51 = c[((x + 5) * n + y) / 8 + 1];

    for (int k = l; k < r; k++) {
        vector a0 = vector{} + a[(x + 0) * n + k];
        t00 += a0 * b[(k * n + y) / 8];
        t01 += a0 * b[(k * n + y) / 8 + 1];

        vector a1 = vector{} + a[(x + 1) * n + k];
        t10 += a1 * b[(k * n + y) / 8];
        t11 += a1 * b[(k * n + y) / 8 + 1];

        vector a2 = vector{} + a[(x + 2) * n + k];
        t20 += a2 * b[(k * n + y) / 8];
        t21 += a2 * b[(k * n + y) / 8 + 1];

        vector a3 = vector{} + a[(x + 3) * n + k];
        t30 += a3 * b[(k * n + y) / 8];
        t31 += a3 * b[(k * n + y) / 8 + 1];

        vector a4 = vector{} + a[(x + 4) * n + k];
        t40 += a4 * b[(k * n + y) / 8];
        t41 += a4 * b[(k * n + y) / 8 + 1];

        vector a5 = vector{} + a[(x + 5) * n + k];
        t50 += a5 * b[(k * n + y) / 8];
        t51 += a5 * b[(k * n + y) / 8 + 1];
    }

    c[((x + 0) * n + y) / 8 + 0] = t00;
    c[((x + 0) * n + y) / 8 + 1] = t01;

    c[((x + 1) * n + y) / 8 + 0] = t10;
    c[((x + 1) * n + y) / 8 + 1] = t11;

    c[((x + 2) * n + y) / 8 + 0] = t20;
    c[((x + 2) * n + y) / 8 + 1] = t21;

    c[((x + 3) * n + y) / 8 + 0] = t30;
    c[((x + 3) * n + y) / 8 + 1] = t31;

    c[((x + 4) * n + y) / 8 + 0] = t40;
    c[((x + 4) * n + y) / 8 + 1] = t41;

    c[((x + 5) * n + y) / 8 + 0] = t50;
    c[((x + 5) * n + y) / 8 + 1] = t51;
}

/*
const int L1 = (1<<15) / 4; // L1 cache is 32K
const int L2 = (1<<19) / 4; // L2 cache is 512K
const int L3 = (1<<23) / 4; // L3 cache is 8M
*/

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
    if (n != 960 && n != 1920 && n != 3840) {
        printf("only n=960,1920 and 3840 are supported.\n");
        exit(-1);
    }
    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;

    const int MAXN = 4000 * 4000;
    alignas(64) static float a[MAXN] = {0}, b[MAXN] = {0}, c[MAXN] = {0};

    for (int i = 0; i < n; i++) {
        memcpy(&a[i * ny], &_a[i * n], 4 * n);
        memcpy(&b[i * ny], &_b[i * n], 4 * n);
    }

    // c[x:x+6][y:y+16] += a[x:x+6][l:r] * b[l:r][y:y+16]

    // load b[i*L1 : (i+1)*L1][y:y+16] into L1 cache and iterate over a
    // when out of L2 cache to hold a, load new strip of b and continue
    // when out of L3 cache to hold b, switch to new segment of a

    // divide b into segments that fit L3, fix a segment
    // divide a into segments that fit L2, fix a segment
    // divide b into segments that fit L1, fix a segment
    // iterate over a

    /*
    // how many columns of b fit in L3
     const int s3 = std::min(L3 / nx / 16 * 16, ny);
    // how many rows of a fit in L2
     const int s2 = std::min(L2 / ny / 6 * 6, nx);
    // how tall a (k x s3) block in b can be to fit in L1
     const int s1 = std::min(L1 / s3, nx);
    */

    // s3 * nx < L3 (doesn't really matter)
    // s2 * ny < L2
    // s1 * s3 < L1
    // s1 -> max

    // const int s1 = std::min(L1 / 16, nx);
    // const int s2 = L2 / ny / 6 * 6;
    // const int s3 = 16;

    const int s3 = 64;
    const int s2 = 120;
    const int s1 = 240;

    /*
    const int u = 96;
    const int s3 = u;
    const int s2 = 2 * u;
    const int s1 = 4 * u;
    */

    // const int t = L1/s3;

    // 1 252 4032
    // std::cerr << s1 << " " << s2 << " " << s3 << std::endl;

    for (int i3 = 0; i3 < ny; i3 += s3)
        // now we are working with b[:][i3:i3+s3]
        for (int i2 = 0; i2 < nx; i2 += s2)
            // now we are working with a[i2:i2+s2][:]
            for (int i1 = 0; i1 < ny; i1 += s1)
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // this equates to updating c[i2:i2+s2][i3:i3+s3]
                // with [l:r] = [i1:i1+s1]
                for (int x = i2; x < i2 + s2; x += 6)
                    for (int y = i3; y < i3 + s3; y += 16)
                        kernel(a, (vector *)b, (vector *)c, x, y, i1, i1 + s1, ny);

    for (int i = 0; i < n; i++)
        memcpy(&_c[i * n], &c[i * ny], 4 * n);
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
