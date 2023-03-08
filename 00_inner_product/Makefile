CXX = g++
CXXFLAGS =-I/home/docker/MPLAPACK/include -Wall -O2 -fopenmp
LDFLAGS_MPFR=-L/home/docker/MPLAPACK/lib -lmpfr -lgmp
LDFLAGS_MPBLAS_MPFR=-L/home/docker/MPLAPACK/lib -lmpblas_mpfr -lmpc -lmpfr -lgmp
CXXFLAGS_MPBLAS_MPFR =-I/home/docker/MPLAPACK/include -I/home/docker/MPLAPACK/include/mplapack -Wall -O2 -fopenmp

TARGETS = inner_product_mpfr_00_naive inner_product_mpfr_01_fma inner_product_mpfr_02_mpreal

SRC00 = inner_product_mpfr_00_naive.cpp
SRC01 = inner_product_mpfr_01_fma.cpp
SRC02 = inner_product_mpfr_02_mpreal.cpp

inner_product_mpfr_00_naive: $(SRC00)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR) 

inner_product_mpfr_01_fma: $(SRC01)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

inner_product_mpfr_02_mpreal: $(SRC02)
	$(CXX) $(CXXFLAGS_MPBLAS_MPFR) $< -o $@ $(LDFLAGS_MPBLAS_MPFR)

all: $(TARGETS)

clean:
	rm -f *.o $(TARGETS) *~

