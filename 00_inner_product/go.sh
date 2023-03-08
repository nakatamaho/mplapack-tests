/usr/bin/time ./inner_product_mpfr_00_naive 100000000 512
/usr/bin/time ./inner_product_mpfr_01_fma 100000000 512
LD_LIBRARY_PATH=/home/docker/MPLAPACK/lib /usr/bin/time ./inner_product_mpfr_02_mpreal 100000000 512
