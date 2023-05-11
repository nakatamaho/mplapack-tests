CXX = g++-11
#CXX = g++-mp-12

MPLAPACK_HOME=/home/docker/MPLAPACK
#MPLAPACK_HOME=/Volumes/Users/maho/MPLAPACK

OPENMPFLAGS = -Wall -O3 -march=native
CXXFLAGS = -I$(MPLAPACK_HOME)/include -fopenmp -Wall -funroll-loops -O3 -march=native
LDFLAGS = -L$(MPLAPACK_HOME)/lib -lgmp -Wl,-rpath,$(MPLAPACK_HOME)/lib
LDFLAGSMPFR = -L$(MPLAPACK_HOME)/lib -lgmp -lmpfr -Wl,-rpath,$(MPLAPACK_HOME)/lib

TARGET = gemm_gmp_10_naive gemm_gmp_12_mpblas gemm_gmp_13_mpblas_openmp gemm_gmp_30_mpblaslike_naive gemm_mpfr_10_naive #gemm_mpfr_11_fma
SRCS = gemm_gmp_10_naive.cpp gemm_gmp_12_mpblas.cpp gemm_gmp_13_mpblas_openmp.cpp gemm_gmp_30_mpblaslike_naive.cpp gemm_mpfr_10_naive.cpp #gemm_mpfr_11_fma.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

gemm_gmp_10_naive: gemm_gmp_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_mpfr_10_naive: gemm_mpfr_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSMPFR)

gemm_mpfr_11_fma: gemm_mpfr_11_fma.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSMPFR)

gemm_gmp_12_mpblas: gemm_gmp_12_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_13_mpblas_openmp: gemm_gmp_13_mpblas_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive: gemm_gmp_30_mpblaslike_naive.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
