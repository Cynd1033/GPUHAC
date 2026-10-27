// run_gpuhac_from_file.cu
// Load a binary CSR graph (as written by snap_to_csr.py) and run gpuhac_naive.
//
// Usage:
//   ./run_gpuhac_from_file <csr_binary_file> [epsilon]
//
// Binary layout (little-endian, matches struct.pack in snap_to_csr.py):
//   int32   n
//   int32   m
//   int32[n+1]  row_ptr
//   int32[m]    col_idx
//   float32[m]  weights

#include "gpuhac_naive.cu"

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cstdint>

std::vector<MergeEvent> gpuhac_naive(
    int n, int m,
    int* h_row_ptr, int* h_col_idx, float* h_weights,
    float epsilon);

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct CSRGraph {
    int n = 0, m = 0;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> weights;
};

static void read_exact(std::ifstream& f, void* buf, size_t bytes, const char* what) {
    f.read(reinterpret_cast<char*>(buf), bytes);
    if (!f) {
        std::cerr << "Error: failed to read " << what
                   << " (expected " << bytes << " bytes)\n";
        std::exit(1);
    }
}

CSRGraph load_csr_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Error: could not open file " << path << "\n";
        std::exit(1);
    }

    CSRGraph G;

    int32_t n32, m32;
    read_exact(f, &n32, sizeof(int32_t), "n");
    read_exact(f, &m32, sizeof(int32_t), "m");
    G.n = static_cast<int>(n32);
    G.m = static_cast<int>(m32);

    if (G.n < 0 || G.m < 0) {
        std::cerr << "Error: parsed negative n or m (n=" << G.n
                   << ", m=" << G.m << ") — file format mismatch?\n";
        std::exit(1);
    }

    G.row_ptr.resize(G.n + 1);
    read_exact(f, G.row_ptr.data(), sizeof(int32_t) * (G.n + 1), "row_ptr");

    G.col_idx.resize(G.m);
    read_exact(f, G.col_idx.data(), sizeof(int32_t) * G.m, "col_idx");

    G.weights.resize(G.m);
    read_exact(f, G.weights.data(), sizeof(float) * G.m, "weights");

    // Check for trailing garbage / truncated file
    char extra;
    f.read(&extra, 1);
    if (!f.eof()) {
        std::cerr << "Warning: file has more data than expected "
                   << "(possible size/format mismatch)\n";
    }

    if (G.row_ptr[G.n] != G.m) {
        std::cerr << "Warning: row_ptr[n] (" << G.row_ptr[G.n]
                   << ") does not match m (" << G.m << ")\n";
    }

    return G;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csr_binary_file> [epsilon]\n";
        return 1;
    }

    std::string path = argv[1];
    float epsilon = (argc >= 3) ? std::atof(argv[2]) : 0.1f;

    std::cout << "Loading CSR graph from: " << path << "\n";

    auto t0 = Clock::now();
    CSRGraph G = load_csr_binary(path);
    auto t1 = Clock::now();
    double load_ms = Ms(t1 - t0).count();

    std::cout << "Loaded: n=" << G.n << " m=" << G.m
               << " (load time: " << std::fixed << std::setprecision(1)
               << load_ms << " ms)\n";
    std::cout << "epsilon=" << epsilon << "\n\n";

    cudaFree(nullptr);
    cudaDeviceSynchronize();

    auto t2 = Clock::now();
    auto dendrogram = gpuhac_naive(G.n, G.m,
                                   G.row_ptr.data(),
                                   G.col_idx.data(),
                                   G.weights.data(),
                                   epsilon);
    cudaDeviceSynchronize();
    auto t3 = Clock::now();
    double run_ms = Ms(t3 - t2).count();

    double edges_per_s = (G.m / 2.0) / (run_ms / 1000.0);

    std::cout << std::left
               << std::setw(12) << "run(ms)"
               << std::setw(10) << "merges"
               << std::setw(16) << "edges/s"
               << "\n"
               << std::string(38, '-') << "\n";
    std::cout << std::left
               << std::setw(12) << std::fixed << std::setprecision(1) << run_ms
               << std::setw(10) << dendrogram.size()
               << std::setw(16) << std::scientific << std::setprecision(2) << edges_per_s
               << "\n";

    return 0;
}
