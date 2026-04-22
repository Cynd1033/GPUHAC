#include "naive_parhac.cuh"

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

// types

struct MergeEvent {
    int   blue;    // cluster absorbed
    int   red;     // cluster that absorbed it
    float weight;  // max weight in this outer round (proxy for merge weight)
};

// helper

static dim3 blocks1d(int n, int threads = 256) {
    return dim3((n + threads - 1) / threads);
}

std::vector<MergeEvent> ParHAC_naive(
    int    n,
    int    m,
    int*   h_row_ptr,
    int*   h_col_idx,
    float* h_weights,
    float  epsilon = 0.1f)
{
    // Copy GraphCSR to device
    GraphCSR G;
    G.n = n;
    G.m = m;
    cudaMalloc(&G.row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&G.col_idx,  m      * sizeof(int));
    cudaMalloc(&G.weights,  m      * sizeof(float));
    cudaMemcpy(G.row_ptr, h_row_ptr, (n + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(G.col_idx, h_col_idx,  m      * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(G.weights, h_weights,  m      * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and initialize ClusterState
    ClusterState CS;
    cudaMalloc(&CS.parent,       n * sizeof(int));
    cudaMalloc(&CS.cluster_size, n * sizeof(int));
    cudaMalloc(&CS.color,        n * sizeof(uint8_t));
    cudaMalloc(&CS.merge_target, n * sizeof(int));
    cudaMalloc(&CS.active,       n * sizeof(bool));

    thrust::sequence(thrust::device, CS.parent, CS.parent + n);              // parent[i] = i
    thrust::fill(thrust::device, CS.cluster_size, CS.cluster_size + n, 1);
    thrust::fill(thrust::device, CS.color,        CS.color + n,        kInactive);
    thrust::fill(thrust::device, CS.merge_target, CS.merge_target + n, -1);
    thrust::fill(thrust::device, CS.active,       CS.active + n,       true);

    // Allocations
    float* d_max_weight;
    int*   d_out_red;
    float* d_out_priority;
    cudaMalloc(&d_max_weight,  sizeof(float));
    cudaMalloc(&d_out_red,      n * sizeof(int));
    cudaMalloc(&d_out_priority, n * sizeof(float));

    // Output
    std::vector<MergeEvent> dendrogram;
    dendrogram.reserve(n - 1);

    const int   threads       = 256;
    const dim3  vertex_blocks = blocks1d(n, threads);
    const float one_plus_eps  = 1.0f + epsilon;

    unsigned long long seed = 42ULL;
    int num_active = n;

    // Outer loop
    while (num_active > 1) {

        // Compute max weight across all active cluster edges
        cudaMemset(d_max_weight, 0, sizeof(float));
        compute_max_weight_kernel<<<vertex_blocks, threads>>>(G, CS, d_max_weight);

        float h_max_weight;
        cudaMemcpy(&h_max_weight, d_max_weight, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_max_weight == 0.0f) break;

        std::cout << "Outer round | max_weight=" << h_max_weight
                  << " num_active=" << num_active << std::endl;

        // Inner loop
        while (true) {

            // Color active vertices
            color_vertices_kernel<<<vertex_blocks, threads>>>(CS, seed++, n);

            // Each blue vertex picks a random red neighbor
            thrust::fill(thrust::device, d_out_red, d_out_red + n, -1);
            build_triples_kernel<<<vertex_blocks, threads>>>(
                G, CS, d_out_red, d_out_priority, seed++, n);

            // Filter to blue vertices that found a red neighbor
            thrust::device_vector<int> d_blue_idx(n);
            int num_valid = thrust::copy_if(
                thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n),
                d_blue_idx.begin(),
                [d_out_red = d_out_red] __device__ (int i) {
                    return d_out_red[i] != -1;
                }
            ) - d_blue_idx.begin();
            d_blue_idx.resize(num_valid);

            if (num_valid == 0) break;

            // Sort blue indices by (chosen red root, priority)
            // This groups all blues targeting the same red together,
            // ordered by priority within each group.
            thrust::sort(
                thrust::device,
                d_blue_idx.begin(),
                d_blue_idx.end(),
                [d_out_red      = d_out_red,
                 d_out_priority = d_out_priority] __device__ (int a, int b) {
                    if (d_out_red[a] != d_out_red[b])
                        return d_out_red[a] < d_out_red[b];
                    return d_out_priority[a] < d_out_priority[b];
                }
            );

            // Compute per-red group offsets via reduce_by_key + exclusive_scan
            // d_sorted_reds[i] = the red root chosen by d_blue_idx[i]
            thrust::device_vector<int> d_sorted_reds(num_valid);
            thrust::transform(
                thrust::device,
                d_blue_idx.begin(), d_blue_idx.end(),
                d_sorted_reds.begin(),
                [d_out_red = d_out_red] __device__ (int i) { return d_out_red[i]; }
            );

            thrust::device_vector<int> d_red_roots(num_valid);
            thrust::device_vector<int> d_red_counts(num_valid);
            int num_red = thrust::reduce_by_key(
                thrust::device,
                d_sorted_reds.begin(), d_sorted_reds.end(),
                thrust::make_constant_iterator(1),
                d_red_roots.begin(),
                d_red_counts.begin()
            ).first - d_red_roots.begin();
            d_red_roots.resize(num_red);
            d_red_counts.resize(num_red);

            // Exclusive scan over counts gives start offset of each red's group
            thrust::device_vector<int> d_red_offsets(num_red + 1);
            thrust::exclusive_scan(
                thrust::device,
                d_red_counts.begin(), d_red_counts.end(),
                d_red_offsets.begin(),
                0
            );
            d_red_offsets[num_red] = num_valid;  // sentinel

            // Prefix select: each red picks the largest prefix of its blues
            // such that total absorbed size <= (1+eps) * red_size
            thrust::fill(thrust::device, CS.merge_target, CS.merge_target + n, -1);
            prefix_select_kernel<<<blocks1d(num_red, threads), threads>>>(
                CS,
                thrust::raw_pointer_cast(d_blue_idx.data()),
                thrust::raw_pointer_cast(d_red_offsets.data()),
                thrust::raw_pointer_cast(d_red_roots.data()),
                one_plus_eps,
                num_red,
                CS.merge_target
            );

            // Count merges — if none, inner loop has converged
            int num_merges = thrust::count_if(
                thrust::device,
                CS.merge_target, CS.merge_target + n,
                [] __device__ (int t) { return t != -1; }
            );
            if (num_merges == 0) break;

            // Apply merges (union-find updates)
            apply_merges_kernel<<<vertex_blocks, threads>>>(CS, CS.merge_target, n);
            num_active -= num_merges;

            // Record merge events for dendrogram
	    std::vector<int>   h_merge_target(n);
	    std::vector<float> h_merge_weight(n);
	    cudaMemcpy(h_merge_target.data(), CS.merge_target, n * sizeof(int),   cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_merge_weight.data(), d_merge_weight,  n * sizeof(float), cudaMemcpyDeviceToHost);
	    for (int i = 0; i < n; i++) {
	      if (h_merge_target[i] != -1) {
		dendrogram.push_back({i, h_merge_target[i], h_merge_weight[i]});
	      }
	    }

            // Reset per-round scratch
            thrust::fill(thrust::device, CS.color,        CS.color + n,        kInactive);
            thrust::fill(thrust::device, CS.merge_target, CS.merge_target + n, -1);
        }
    }

    // free memory
    cudaFree(G.row_ptr);
    cudaFree(G.col_idx);
    cudaFree(G.weights);
    cudaFree(CS.parent);
    cudaFree(CS.cluster_size);
    cudaFree(CS.color);
    cudaFree(CS.merge_target);
    cudaFree(CS.active);
    cudaFree(d_max_weight);
    cudaFree(d_out_red);
    cudaFree(d_out_priority);

    return dendrogram;
}
