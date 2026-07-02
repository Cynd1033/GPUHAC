CXX      = nvcc
CXXFLAGS = -O2 -std=c++17 --extended-lambda

all: bench_gpuhac

bench_gpuhac: bench_gpuhac.cu gpuhac_naive.cu gpuhac_naive.cuh gpuhac_naive_types.h
	$(CXX) $(CXXFLAGS) -o $@ bench_gpuhac.cu

clean:
	rm -f bench_gpuhac
