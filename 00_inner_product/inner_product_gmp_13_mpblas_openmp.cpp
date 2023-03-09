#include <iostream>
#include <chrono>
#include <gmp.h>
#include <gmpxx.h>

gmp_randstate_t state;

mpf_class Rdot(long n, mpf_class *dx, long incx, mpf_class *dy, long incy) {
    long ix = 0;
    long iy = 0;
    long i;
    mpf_class temp, templ;

    temp = 0.0;

    if (incx < 0)
        ix = (-n + 1) * incx;
    if (incy < 0)
        iy = (-n + 1) * incy;

    temp = 0.0;
    if (incx == 1 && incy == 1) {
// no reduction for multiple precision
#ifdef _OPENMP
#pragma omp parallel private(i, templ) shared(temp, dx, dy, n)
#endif
        {
            templ = 0.0;
#ifdef _OPENMP
#pragma omp for
#endif
            for (i = 0; i < n; i++) {
                templ += dx[i] * dy[i];
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            temp += templ;
        }
    } else {
        for (i = 0; i < n; i++) {
            temp += dx[ix] * dy[iy];
            ix = ix + incx;
            iy = iy + incy;
        }
    }
    return temp;
}

void init_mpf_vec(mpf_t *vec, int n, int prec) {
    for (int i = 0; i < n; i++) {
        mpf_init2(vec[i], prec);
        mpf_urandomb(vec[i], state, prec);
    }
}

void clear_mpf_vec(mpf_t *vec, int n) {
    for (int i = 0; i < n; i++) {
        mpf_clear(vec[i]);
    }
}

int main(int argc, char **argv) {
    gmp_randinit_default(state);
    gmp_randseed_ui(state, 42);

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <vector size> <precision>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int prec = std::atoi(argv[2]);
    mpf_set_default_prec(prec);

    mpf_t *vec1 = new mpf_t[N];
    mpf_t *vec2 = new mpf_t[N];
    mpf_t tmp, dot_product;

    mpf_init2(dot_product, prec);
    mpf_init2(tmp, prec);
    init_mpf_vec(vec1, N, prec);
    init_mpf_vec(vec2, N, prec);

    mpf_class *vec1_mpf_class = new mpf_class[N];
    mpf_class *vec2_mpf_class = new mpf_class[N];
    mpf_class ans;

    for (int i = 0; i < N; i++) {
        vec1_mpf_class[i] = mpf_class(vec1[i]);
        vec2_mpf_class[i] = mpf_class(vec2[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    ans = Rdot(N, vec1_mpf_class, 1, vec2_mpf_class, 1);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    gmp_printf("%.128Ff", ans);
    std::cout << std::endl;

    clear_mpf_vec(vec1, N);
    clear_mpf_vec(vec2, N);
    mpf_clear(tmp);
    mpf_clear(dot_product);
    delete[] vec1;
    delete[] vec2;

    return 0;
}
