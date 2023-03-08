CXX = g++

MPLAPACK_HOME=/home/docker/MPLAPACK

LDFLAGS_MPFR=-L$(MPLAPACK_HOME)/lib -lmpfr -lgmp
LDFLAGS_GMP=-L$(MPLAPACK_HOME)/lib -lmpfr -lgmp
LDFLAGS_MPBLAS_MPFR=-L$(MPLAPACK_HOME)/lib -lmpblas_mpfr -lmpc -lmpfr -lgmp
LDFLAGS_MPBLAS_GMP=-L$(MPLAPACK_HOME)/lib -lmpblas_gmp -lgmp

CXXFLAGS =-I$(MPLAPACK_HOME)/include -Wall -O2 -fopenmp
CXXFLAGS_MPBLAS_MPFR =-I$(MPLAPACK_HOME)/include -I$(MPLAPACK_HOME)/include/mplapack -Wall -O2 -fopenmp
CXXFLAGS_MPBLAS_GMP =-I$(MPLAPACK_HOME)/include -I$(MPLAPACK_HOME)/include/mplapack -Wall -O2 -fopenmp

TARGETS = inner_product_mpfr_00_naive inner_product_mpfr_01_fma inner_product_mpfr_02_mpblas \
          inner_product_gmp_10_naive inner_product_gmp_12_mpblas

SRC00 = inner_product_mpfr_00_naive.cpp
SRC01 = inner_product_mpfr_01_fma.cpp
SRC02 = inner_product_mpfr_02_mpblas.cpp
SRC10 = inner_product_gmp_10_naive.cpp
SRC12 = inner_product_gmp_12_mpblas.cpp

inner_product_mpfr_00_naive: $(SRC00)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_01_fma: $(SRC01)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_02_mpblas: $(SRC02)
	$(CXX) $(CXXFLAGS_MPBLAS_MPFR) $< -o $@ $(LDFLAGS_MPBLAS_MPFR)

inner_product_gmp_10_naive: $(SRC10)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_GMP)

inner_product_gmp_12_mpblas: $(SRC12)
	$(CXX) $(CXXFLAGS_MPBLAS_GMP) $< -o $@ $(LDFLAGS_MPBLAS_GMP)

all: $(TARGETS)

clean:
	rm -f *.o $(TARGETS) *~

