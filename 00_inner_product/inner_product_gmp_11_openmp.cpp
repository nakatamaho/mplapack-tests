#include <iostream>
#include <chrono>
#include <gmp.h>

gmp_randstate_t state;

void Rdot(long n, mpf_t *dx, long incx, mpf_t *dy, long incy, mpf_t *ans, mp_bitcnt_t prec) {
    long ix = 0;
    long iy = 0;
    long i;
    mpf_t templ, templl;

    if (incx < 0)
        ix = (-n + 1) * incx;
    if (incy < 0)
        iy = (-n + 1) * incy;

    if (incx == 1 && incy == 1) {
        mpf_set_d(*ans, 0.0);
// no reduction for multiple precision
#ifdef _OPENMP
#pragma omp parallel private(i, templ, templl) shared(ans, dx, dy, n)
#endif
        {
            mpf_init2(templ, prec);
            mpf_init2(templl, prec);
            mpf_set_d(templ, 0.0);
            mpf_set_d(templl, 0.0);
#ifdef _OPENMP
#pragma omp for
#endif
            for (i = 0; i < n; i++) {
                mpf_mul(templl, dx[i], dy[i]);
                mpf_add(templ, templ, templl);
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            mpf_add(*ans, *ans, templ);
            mpf_clear(templ);
            mpf_clear(templl);
        }
    } else {
      printf("Not supported, exitting\n");
      exit(-1);
    }
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
    mpf_t ans, dot_product;

    mpf_init2(dot_product, prec);
    mpf_init2(ans, prec);
    init_mpf_vec(vec1, N, prec);
    init_mpf_vec(vec2, N, prec);

    auto start = std::chrono::high_resolution_clock::now();
    Rdot(N, vec1, 1, vec2, 1, &ans, prec);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    gmp_printf("%.128Ff", ans);
    std::cout << std::endl;

    clear_mpf_vec(vec1, N);
    clear_mpf_vec(vec2, N);
    mpf_clear(dot_product);
    delete[] vec1;
    delete[] vec2;

    return 0;
}
