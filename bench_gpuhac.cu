// bench_gpuhac.cu
// Standalone benchmark for gpuhac_naive using planted partition graphs
//
//
// Usage:
//   ./bench_gpuhac            # runs default sweep
//   ./bench_gpuhac 10000 20   # single run: n=10000, k=20 clusters

#include "gpuhac_naive.cu"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// Forward declaration 

std::vector<MergeEvent> gpuhac_naive(
    int n, int m,
    int* h_row_ptr, int* h_col_idx, float* h_weights,
    float epsilon);

// Planted partition graph generator
//
// k clusters of size n/k (last cluster absorbs remainder).
// Intra-cluster edges: probability p_in,  weight ~ U(w_in_lo,  w_in_hi)
// Inter-cluster edges: probability p_out, weight ~ U(w_out_lo, w_out_hi)
// Graph is undirected; CSR stores both directions.

struct CSRGraph {
    int n, m;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> weights;
};

CSRGraph make_planted_partition(
    int    n,
    int    k,
    float  p_in   = 0.3f,
    float  p_out  = 0.02f,
    float  w_in_lo  = 0.5f, float w_in_hi  = 1.0f,
    float  w_out_lo = 0.0f, float w_out_hi = 0.2f,
    uint64_t seed = 1234)
{
    assert(k >= 1 && k <= n);

    std::mt19937_64           rng(seed);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::uniform_real_distribution<float> u_in (w_in_lo,  w_in_hi);
    std::uniform_real_distribution<float> u_out(w_out_lo, w_out_hi);

    // Assign cluster membership
    std::vector<int> cluster(n);
    for (int i = 0; i < n; i++) cluster[i] = i / (n / k);  // last cluster may be larger

    // Build adjacency list (upper triangle, then mirror)
    std::vector<std::vector<std::pair<int,float>>> adj(n);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            bool same = (cluster[i] == cluster[j]);
            float p   = same ? p_in : p_out;
            if (u01(rng) < p) {
                float w = same ? u_in(rng) : u_out(rng);
                adj[i].push_back({j, w});
                adj[j].push_back({i, w});
            }
        }
    }

    // Build CSR
    CSRGraph G;
    G.n = n;
    G.row_ptr.resize(n + 1);
    int total = 0;
    for (int i = 0; i < n; i++) {
        G.row_ptr[i] = total;
        total += (int)adj[i].size();
    }
    G.row_ptr[n] = total;
    G.m = total;

    G.col_idx.resize(total);
    G.weights.resize(total);
    for (int i = 0; i < n; i++) {
        int base = G.row_ptr[i];
        for (int e = 0; e < (int)adj[i].size(); e++) {
            G.col_idx[base + e] = adj[i][e].first;
            G.weights[base + e] = adj[i][e].second;
        }
    }

    return G;
}

// timing helper 

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// single benchmark run

struct BenchResult {
    int    n, m, k;
    double gen_ms;      // graph generation time
    double run_ms;      // ParHAC_naive wall time (after sync)
    int    merges;      // dendrogram events (expect n-1 for connected graph)
    double edges_per_s;
};

BenchResult run_one(int n, int k, float epsilon, uint64_t seed) {
    BenchResult r;
    r.n = n;
    r.k = k;

    // Generate graph
    auto t0 = Clock::now();
    CSRGraph G = make_planted_partition(n, k, /*p_in=*/0.3f, /*p_out=*/0.02f,
                                        0.5f, 1.0f, 0.0f, 0.2f, seed);
    auto t1 = Clock::now();
    r.m      = G.m;
    r.gen_ms = Ms(t1 - t0).count();

    // Warmup: force CUDA context init before timing
    cudaFree(nullptr);
    cudaDeviceSynchronize();

    // Timed run
    auto t2 = Clock::now();
    auto dendrogram = gpuhac_naive(n, G.m,
                                   G.row_ptr.data(),
                                   G.col_idx.data(),
                                   G.weights.data(),
                                   epsilon);
    cudaDeviceSynchronize();
    auto t3 = Clock::now();

    r.run_ms      = Ms(t3 - t2).count();
    r.merges      = (int)dendrogram.size();
    r.edges_per_s = (G.m / 2.0) / (r.run_ms / 1000.0);  // undirected edges/s

    return r;
}

// print helpers
static void print_header() {
    std::cout
        << std::left
        << std::setw(8)  << "n"
        << std::setw(6)  << "k"
        << std::setw(12) << "edges"
        << std::setw(12) << "gen(ms)"
        << std::setw(12) << "run(ms)"
        << std::setw(10) << "merges"
        << std::setw(16) << "edges/s"
        << "\n"
        << std::string(76, '-') << "\n";
}

static void print_row(const BenchResult& r) {
    std::cout
        << std::left
        << std::setw(8)  << r.n
        << std::setw(6)  << r.k
        << std::setw(12) << (r.m / 2)       // undirected
        << std::setw(12) << std::fixed << std::setprecision(1) << r.gen_ms
        << std::setw(12) << std::fixed << std::setprecision(1) << r.run_ms
        << std::setw(10) << r.merges
        << std::setw(16) << std::scientific << std::setprecision(2) << r.edges_per_s
        << "\n";
}

// Main
int main(int argc, char** argv) {
    const float epsilon = 0.1f;

    // Single-run mode: ./bench_gpuhac <n> <k>
    if (argc == 3) {
        int n = std::atoi(argv[1]);
        int k = std::atoi(argv[2]);
        std::cout << "Single run: n=" << n << " k=" << k
                  << " eps=" << epsilon << "\n\n";
        print_header();
        print_row(run_one(n, k, epsilon, /*seed=*/42));
        return 0;
    }

    // Sweep mode
    // Configs: {n, k} — k scales with n so cluster size stays ~500
    struct Cfg { int n, k; };
    std::vector<Cfg> sweep = {
        {  1000,   2},
        {  1000,  10},
        {  5000,  10},
        {  5000,  25},
        { 10000,  20},
        { 10000,  50},
        { 20000,  40},
        { 50000, 100},
    };

    std::cout << "ParHAC_naive benchmark — planted partition graphs\n"
              << "epsilon=" << epsilon << "\n\n";
    print_header();

    uint64_t seed = 1;
    for (auto& cfg : sweep) {
        BenchResult r = run_one(cfg.n, cfg.k, epsilon, seed++);
        print_row(r);
    }

    return 0;
}
