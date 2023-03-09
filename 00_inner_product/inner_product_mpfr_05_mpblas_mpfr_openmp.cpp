#include <iostream>
#include <chrono>
#include <mpreal.h>

gmp_randstate_t state;

mp_rnd_t mpfr::mpreal::default_rnd = MPFR_RNDN; // must be initialized at mpblas/reference/mplapackinit.cpp
mp_prec_t mpfr::mpreal::default_prec = 512;
int mpfr::mpreal::default_base = 2;
int mpfr::mpreal::double_bits = -1;

mpfr::mpreal Rdot(long n, mpfr::mpreal *dx, long incx, mpfr::mpreal *dy, long incy) {
    long ix = 0;
    long iy = 0;
    long i;
    mpfr::mpreal ans;
    mpfr_t templ;

    temp = 0.0;

    if (incx < 0)
        ix = (-n + 1) * incx;
    if (incy < 0)
        iy = (-n + 1) * incy;

    temp = 0.0;
    if (incx == 1 && incy == 1) {
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
                mpfr_fma(templ, mpfr_ptr(dx[i]), mpfr_ptr(dy[i]), templ, MPFR_RNDN);
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            mpfr_add(mpfr_ptr(ans), mpfr_ptr(ans), templ, MPFR_RNDN);
            mpfr_clear(templ);
        }
    } else {
        printf("Not supported, exitting\n");
        exit(-1);
    }
    return ans;
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
    mpfr::mpreal::default_prec = prec;

    mpfr_t *vec1 = new mpfr_t[N];
    mpfr_t *vec2 = new mpfr_t[N];
    mpfr_t dot_product;

    mpfr_init2(dot_product, prec);
    mpfr::mpreal::default_prec = prec;

    init_mpfr_vec(vec1, N, prec);
    init_mpfr_vec(vec2, N, prec);
    mpfr::mpreal *vec1_mpreal = new mpfr::mpreal[N];
    mpfr::mpreal *vec2_mpreal = new mpfr::mpreal[N];
    mpfr::mpreal ans;

    for (int i = 0; i < N; i++) {
        vec1_mpreal[i] = vec1[i];
        vec2_mpreal[i] = vec2[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    ans = Rdot(N, vec1_mpreal, 1, vec2_mpreal, 1);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    mpfr_printf("%.128Rf", mpfr_ptr(ans));
    std::cout << std::endl;

    clear_mpfr_vec(vec1, N);
    clear_mpfr_vec(vec2, N);
    mpfr_clear(dot_product);
    delete[] vec1_mpreal;
    delete[] vec2_mpreal;
    delete[] vec1;
    delete[] vec2;

    return 0;
}
