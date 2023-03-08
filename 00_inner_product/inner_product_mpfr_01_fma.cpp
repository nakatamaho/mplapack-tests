#include <iostream>
#include <chrono>
#include <mpfr.h>

gmp_randstate_t state;

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
    mpfr_t dot_product;

    mpfr_init2(dot_product, prec);
    init_mpfr_vec(vec1, N, prec);
    init_mpfr_vec(vec2, N, prec);

    auto start = std::chrono::high_resolution_clock::now();
    mpfr_set_d(dot_product, 0.0, MPFR_RNDN);
    for (int i = 0; i < N; i++) {
        mpfr_fma(dot_product, vec1[i], vec2[i], dot_product, MPFR_RNDN);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    mpfr_printf("%.128Rf", dot_product);
    std::cout << std::endl;

    clear_mpfr_vec(vec1, N);
    clear_mpfr_vec(vec2, N);
    mpfr_clear(dot_product);
    delete[] vec1;
    delete[] vec2;

    return 0;
}
