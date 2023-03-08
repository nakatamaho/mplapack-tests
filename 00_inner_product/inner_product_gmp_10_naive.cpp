#include <iostream>
#include <chrono>
#include <gmp.h>

gmp_randstate_t state;

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

    mpf_t *vec1 = new mpf_t[N];
    mpf_t *vec2 = new mpf_t[N];
    mpf_t tmp, dot_product;

    mpf_init2(dot_product, prec);
    mpf_init2(tmp, prec);
    init_mpf_vec(vec1, N, prec);
    init_mpf_vec(vec2, N, prec);

    mpf_set_d(dot_product, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        mpf_mul(tmp, vec1[i], vec2[i]);
        mpf_add(dot_product, dot_product, tmp);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s" << std::endl;

    std::cout << "Dot product: ";
    gmp_printf("%.128Ff", dot_product);
    std::cout << std::endl;

    clear_mpf_vec(vec1, N);
    clear_mpf_vec(vec2, N);
    mpf_clear(tmp);
    mpf_clear(dot_product);
    delete[] vec1;
    delete[] vec2;

    return 0;
}
