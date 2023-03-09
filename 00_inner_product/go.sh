export LD_LIBRARY_PATH=/home/docker/MPLAPACK/lib
#/usr/bin/time ./inner_product_gmp_10_naive          100000000 512
/usr/bin/time ./inner_product_gmp_12_mpblas         100000000 512
/usr/bin/time ./inner_product_gmp_13_mpblas_openmp  100000000 512

#/usr/bin/time ./inner_product_mpfr_00_naive         100000000 512
#/usr/bin/time ./inner_product_mpfr_01_fma           100000000 512
#/usr/bin/time ./inner_product_mpfr_02_mpblas        100000000 512
