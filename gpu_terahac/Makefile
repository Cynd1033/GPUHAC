CXX      = nvcc
CXXFLAGS = -O2 -std=c++17 --extended-lambda

all: bench_gpuhac run_gpuhac_from_file

bench_gpuhac: bench_gpuhac.cu gpuhac_naive.cu gpuhac_naive.cuh gpuhac_naive_types.h
	$(CXX) $(CXXFLAGS) -o $@ bench_gpuhac.cu

run_gpuhac_from_file: run_gpuhac_from_file.cu gpuhac_naive.cu gpuhac_naive.cuh gpuhac_naive_types.h
	$(CXX) $(CXXFLAGS) -o $@ run_gpuhac_from_file.cu

clean:
	rm -f 
