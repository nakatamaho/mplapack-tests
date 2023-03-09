#include <iostream>
#include <chrono>
#include <mpfr.h>

gmp_randstate_t state;

void Rdot(long n, mpfr_t *dx, long incx, mpfr_t *dy, long incy, mpfr_t *ans, mp_bitcnt_t prec) {
    long ix = 0;
    long iy = 0;
    long i;
    mpfr_t templ;

    if (incx < 0)
        ix = (-n + 1) * incx;
    if (incy < 0)
        iy = (-n + 1) * incy;

    if (incx == 1 && incy == 1) {
        mpfr_set_d(*ans, 0.0, MPFR_RNDN);
// no reduction for multiple precision
#ifdef _OPENMP
#pragma omp parallel private(i, templ) shared(ans, dx, dy, n)
#endif
        {
            mpfr_init2(templ, prec);
            mpfr_set_d(templ, 0.0, MPFR_RNDN);
#ifdef _OPENMP
#pragma omp for
#endif
            for (i = 0; i < n; i++) {
                mpfr_fma(templ, dx[i], dy[i], templ, MPFR_RNDN);
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            mpfr_add(*ans, *ans, templ, MPFR_RNDN);
            mpfr_clear(templ);
        }
    } else {
        printf("Not supported, exitting\n");
        exit(-1);
    }
}

void init_mpfr_vec(mpfr_t *vec, int n, int prec) {
    for (int i = 0; i < n; i++) {
        mpfr_init2(vec[i], prec);
        mpfr_urandom(vec[i], state, MPFR_RNDN);
    }
}

void clear_mpfr_vec(mpfr_t *vec, int n) {
    for (int i = 0; i < n; i++) {
        mpfr_clear(vec[i]);
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

    mpfr_t *vec1 = new mpfr_t[N];
    mpfr_t *vec2 = new mpfr_t[N];
    mpfr_t tmp, dot_product;

    mpfr_init2(dot_product, prec);
    mpfr_init2(tmp, prec);
    init_mpfr_vec(vec1, N, prec);
    init_mpfr_vec(vec2, N, prec);

    mpfr_set_d(dot_product, 0.0, MPFR_RNDN);

    auto start = std::chrono::high_resolution_clock::now();
    Rdot(N, vec1, 1, vec2, 1, &dot_product, prec);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    mpfr_printf("%.128Rf", dot_product);
    std::cout << std::endl;

    clear_mpfr_vec(vec1, N);
    clear_mpfr_vec(vec2, N);
    mpfr_clear(tmp);
    mpfr_clear(dot_product);
    delete[] vec1;
    delete[] vec2;

    return 0;
}
