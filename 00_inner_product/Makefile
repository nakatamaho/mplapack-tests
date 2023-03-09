CXX = g++-11
#CXX = g++-mp-12

MPLAPACK_HOME=/home/docker/MPLAPACK
#MPLAPACK_HOME=/Volumes/Users/maho/MPLAPACK

LDFLAGS_MPFR=-L$(MPLAPACK_HOME)/lib -lmpfr -lgmp
LDFLAGS_GMP=-L$(MPLAPACK_HOME)/lib -lgmpxx -lgmp
LDFLAGS_MPBLAS_MPFR=-L$(MPLAPACK_HOME)/lib -lmpblas_mpfr -lmpc -lmpfr -lgmp
LDFLAGS_MPBLAS_GMP=-L$(MPLAPACK_HOME)/lib -lmpblas_gmp -lgmp

CXXFLAGS =-I$(MPLAPACK_HOME)/include -I$(MPLAPACK_HOME)/include/mplapack -Wall -march=native -O3 -fopenmp

TARGETS = inner_product_mpfr_00_naive inner_product_mpfr_01_fma inner_product_mpfr_02_mpblas inner_product_mpfr_03_openmp \
          inner_product_gmp_10_naive inner_product_gmp_11_openmp inner_product_gmp_12_mpblas inner_product_gmp_13_mpblas_openmp

SRC00 = inner_product_mpfr_00_naive.cpp
SRC01 = inner_product_mpfr_01_fma.cpp
SRC02 = inner_product_mpfr_02_mpblas.cpp
SRC03 = inner_product_mpfr_03_openmp.cpp
SRC10 = inner_product_gmp_10_naive.cpp
SRC11 = inner_product_gmp_11_openmp.cpp
SRC12 = inner_product_gmp_12_mpblas.cpp
SRC13 = inner_product_gmp_13_mpblas_openmp.cpp

inner_product_mpfr_00_naive: $(SRC00)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_01_fma: $(SRC01)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_02_mpblas: $(SRC02)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_03_openmp: $(SRC03)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_gmp_10_naive: $(SRC10)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_GMP)

inner_product_gmp_11_openmp: $(SRC11)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_GMP)

inner_product_gmp_12_mpblas: $(SRC12)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_GMP)

inner_product_gmp_13_mpblas_openmp: $(SRC13)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_GMP)

all: $(TARGETS)

clean:
	rm -f *.o $(TARGETS) *~

