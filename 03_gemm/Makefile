CXX = g++-11
#CXX = g++-mp-12

MPLAPACK_HOME=/home/docker/MPLAPACK
#MPLAPACK_HOME=/Volumes/Users/maho/MPLAPACK

OPENMPFLAGS = -Wall -O2 -march=native
CXXFLAGS = -I$(MPLAPACK_HOME)/include -fopenmp -Wall -funroll-loops -O2 -march=native
LDFLAGS = -L$(MPLAPACK_HOME)/lib -lgmp -Wl,-rpath,$(MPLAPACK_HOME)/lib
LDFLAGSMPFR = -L$(MPLAPACK_HOME)/lib -lgmp -lmpfr -Wl,-rpath,$(MPLAPACK_HOME)/lib

TARGET = gemm_gmp_10_naive \
gemm_gmp_11_jli_2 \
gemm_gmp_11_jli_3 \
gemm_gmp_20_mpblas gemm_gmp_21_mpblas_openmp \
gemm_gmp_30_mpblaslike_naive_ijl \
gemm_gmp_30_mpblaslike_naive_jil \
gemm_gmp_30_mpblaslike_naive_jli \
gemm_gmp_30_mpblaslike_naive_jli_2 \
gemm_gmp_31_mpblaslike_blocking gemm_gmp_32_mpblaslike_blocking_openmp \
gemm_mpfr_10_naive #gemm_mpfr_11_fma

SRCS = gemm_gmp_10_naive.cpp \
gemm_gmp_11_jli_2.cpp \
gemm_gmp_11_jli_3.cpp \
gemm_gmp_20_mpblas.cpp gemm_gmp_21_mpblas_openmp.cpp \
gemm_gmp_30_mpblaslike_naive_ijl.cpp \
gemm_gmp_30_mpblaslike_naive_jil.cpp \
gemm_gmp_30_mpblaslike_naive_jli.cpp \
gemm_gmp_30_mpblaslike_naive_jli_2.cpp \
gemm_gmp_31_mpblaslike_blocking.cpp gemm_gmp_32_mpblaslike_blocking_openmp.cpp \
gemm_mpfr_10_naive.cpp #gemm_mpfr_11_fma.cpp

OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

gemm_gmp_10_naive: gemm_gmp_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_11_jli_2: gemm_gmp_11_jli_2.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_11_jli_3: gemm_gmp_11_jli_3.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_mpfr_10_naive: gemm_mpfr_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSMPFR)

gemm_mpfr_11_fma: gemm_mpfr_11_fma.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSMPFR)

gemm_gmp_20_mpblas: gemm_gmp_20_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_21_mpblas_openmp: gemm_gmp_21_mpblas_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive_ijl: gemm_gmp_30_mpblaslike_naive_ijl.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive_jil: gemm_gmp_30_mpblaslike_naive_jil.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive_jli: gemm_gmp_30_mpblaslike_naive_jli.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive_jli_2: gemm_gmp_30_mpblaslike_naive_jli_2.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_31_mpblaslike_blocking: gemm_gmp_31_mpblaslike_blocking.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_32_mpblaslike_blocking_openmp: gemm_gmp_32_mpblaslike_blocking_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
