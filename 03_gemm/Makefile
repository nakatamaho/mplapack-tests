CXX = g++-11
CXXFLAGS = -Wall -O3 -march=native
LDFLAGS = -lgmp

TARGET = gemm_gmp_10_naive gemm_gmp_12_mpblas
SRCS = gemm_gmp_10_naive.cpp gemm_gmp_12_mpblas.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

gemm_gmp_10_naive: gemm_gmp_10_naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gemm_gmp_12_mpblas: gemm_gmp_12_mpblas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgmpxx

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
