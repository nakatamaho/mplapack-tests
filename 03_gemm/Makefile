CXX = g++-11
#CXX = g++-mp-12

MPLAPACK_HOME=/home/docker/MPLAPACK
#MPLAPACK_HOME=/Volumes/Users/maho/MPLAPACK

OPENMPFLAGS = -Wall -O2 -march=native
CXXFLAGS = -I$(MPLAPACK_HOME)/include -fopenmp -Wall -funroll-loops -O3 -march=native -ffast-math
LDFLAGS = -L$(MPLAPACK_HOME)/lib -lgmp -Wl,-rpath,$(MPLAPACK_HOME)/lib
LDFLAGSMPFR = -L$(MPLAPACK_HOME)/lib -lgmp -lmpfr -Wl,-rpath,$(MPLAPACK_HOME)/lib
LDFLAGSDOUBLE = #-L$(MPLAPACK_HOME)/lib -lgmp -lmpfr -Wl,-rpath,$(MPLAPACK_HOME)/lib

TARGET = \
gemm_float_00_mpblas \
gemm_float_10_naive \
gemm_float_11_transposition \
gemm_float_12_vectorization \
gemm_float_13_kernel \
gemm_float_14_blocking \
gemm_float_15_unroll \
gemm_double_00_mpblas \
gemm_double_10_naive \
gemm_double_11_transposition \
gemm_double_12_vectorization \
gemm_double_13_kernel \
gemm_double_14_blocking \
gemm_gmp_10_naive \
gemm_gmp_11_jli_2 \
gemm_gmp_11_jli_2_openmp \
gemm_gmp_11_jli_3 \
gemm_gmp_20_mpblas gemm_gmp_21_mpblas_openmp \
gemm_gmp_30_mpblaslike_naive_ijl \
gemm_gmp_30_mpblaslike_naive_jil \
gemm_gmp_30_mpblaslike_naive_jli \
gemm_gmp_30_mpblaslike_naive_jli_2 \
gemm_gmp_30_mpblaslike_naive_jli_2_openmp \
gemm_gmp_30_mpblaslike_naive_jli_3 \
gemm_gmp_31_mpblaslike_blocking gemm_gmp_32_mpblaslike_blocking_openmp \
gemm_mpfr_10_naive #gemm_mpfr_11_fma

SRCS = \
gemm_float_00_mpblas.cpp \
gemm_float_10_naive.cpp \
gemm_float_11_transposition.cpp \
gemm_float_12_vectorization.cpp \
gemm_float_13_kernel.cpp \
gemm_float_14_blocking.cpp \
gemm_float_15_unroll.cpp \
gemm_double_00_mpblas.cpp \
gemm_double_10_naive.cpp \
gemm_double_11_transposition.cpp \
gemm_double_12_vectorization.cpp \
gemm_double_13_kernel.cpp \
gemm_double_14_blocking.cpp \
gemm_gmp_10_naive.cpp \
gemm_gmp_11_jli_2.cpp \
gemm_gmp_11_jli_2_openmp.cpp \
gemm_gmp_11_jli_3.cpp \
gemm_gmp_20_mpblas.cpp gemm_gmp_21_mpblas_openmp.cpp \
gemm_gmp_30_mpblaslike_naive_ijl.cpp \
gemm_gmp_30_mpblaslike_naive_jil.cpp \
gemm_gmp_30_mpblaslike_naive_jli.cpp \
gemm_gmp_30_mpblaslike_naive_jli_2.cpp \
gemm_gmp_30_mpblaslike_naive_jli_2_openmp.cpp \
gemm_gmp_30_mpblaslike_naive_jli_3.cpp \
gemm_gmp_31_mpblaslike_blocking.cpp gemm_gmp_32_mpblaslike_blocking_openmp.cpp \
gemm_mpfr_10_naive.cpp #gemm_mpfr_11_fma.cpp

OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

gemm_float_00_mpblas: gemm_float_00_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_10_naive: gemm_float_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_11_transposition: gemm_float_11_transposition.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_12_vectorization: gemm_float_12_vectorization.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_13_kernel: gemm_float_13_kernel.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_14_blocking: gemm_float_14_blocking.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_float_15_unroll: gemm_float_15_unroll.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_double_00_mpblas: gemm_double_00_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_double_10_naive: gemm_double_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_double_11_transposition: gemm_double_11_transposition.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_double_12_vectorization: gemm_double_12_vectorization.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_double_13_kernel: gemm_double_13_kernel.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_double_14_blocking: gemm_double_14_blocking.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGSDOUBLE)

gemm_gmp_10_naive: gemm_gmp_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_11_jli_2: gemm_gmp_11_jli_2.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_11_jli_2_openmp: gemm_gmp_11_jli_2_openmp.o
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

gemm_gmp_30_mpblaslike_naive_jli_3: gemm_gmp_30_mpblaslike_naive_jli_3.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_30_mpblaslike_naive_jli_2_openmp: gemm_gmp_30_mpblaslike_naive_jli_2_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_31_mpblaslike_blocking: gemm_gmp_31_mpblaslike_blocking.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

gemm_gmp_32_mpblaslike_blocking_openmp: gemm_gmp_32_mpblaslike_blocking_openmp.o
	$(CXX) $(OPENMPFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
