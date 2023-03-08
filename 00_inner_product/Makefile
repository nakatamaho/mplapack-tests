CXX = g++
CXXFLAGS =-I/home/docker/MPLAPACK/include -Wall -O2 -fopenmp
LDFLAGS_MPFR=-L/home/docker/MPLAPACK/lib -lmpfr -lgmp
TARGETS = inner_product_mpfr_00_naive inner_product_mpfr_01_fma

SRC01 = inner_product_mpfr_00_naive.cpp
SRC02 = inner_product_mpfr_01_fma.cpp

inner_product_mpfr_00_naive: $(SRC01)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR) 

inner_product_mpfr_01_fma: $(SRC02)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS_MPFR)

all: $(TARGETS)

clean:
	rm -f *.o $(TARGETS) *~

