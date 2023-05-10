CXX = g++-11
CXXFLAGS = -Wall -O2
LDFLAGS = -lgmp

TARGET = gemm_gmp_10_naive
SRCS = gemm_gmp_10_naive.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *~
