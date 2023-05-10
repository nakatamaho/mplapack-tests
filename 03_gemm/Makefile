CXX = g++-11
OPENMPFLAGS = -Wall -O3 -march=native
CXXFLAGS = -fopenmp -Wall -O3 -march=native
LDFLAGS = -lgmp

TARGET = gemm_gmp_10_naive gemm_gmp_12_mpblas gemm_gmp_13_mpblas_openmp
SRCS = gemm_gmp_10_naive.cpp gemm_gmp_12_mpblas.cpp gemm_gmp_13_mpblas_openmp.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

gemm_gmp_10_naive: gemm_gmp_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_12_mpblas: gemm_gmp_12_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_13_mpblas_openmp: gemm_gmp_13_mpblas_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
