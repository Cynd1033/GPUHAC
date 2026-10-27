#include "cuda.h"
#include <cstdint>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>>

struct GraphCSR {
  int64_t    n;         // num vertices
  int64_t    m;         // num edges (directed, so 2x undirected count)
  int64_t*   row_ptr;   // length n+1
  int64_t*   col_idx;   // length m
  float* weights;   // length m
};

struct PartitionCSR {
  int64_t* partition_offsets;
  int64_t* vertices;
};

struct MergeEvent {
  int64_t a; // vertex ids
  int64_t b;
  float sim;
}

__global__ void partition_nodes(int64_t n, int num_parts, uint_64_t seed, int32_t* partition_id){
  int64_t idx = blockIdx * (int64_t)blockDim.x + threadIdx.x;
  if (idx >= n) return;

  curandStatePhilox4_32_10_t gen;
  curand_init(seed, idx, 0, &gen);

  partition_id[idx] = curand(&gen) % num_parts;
}

void get_partition_offsets(int_64t n, int num_parts, int32_t* d_partition_id, int64_t* d_partition_offsets, int64_t* d_vertices){
  // takes in partitioned graph, sorts nodes by partition assignment p in 0, . . ., num_parts
  thrust::device_ptr<int32_t> keys(d_partition_id);
  thrust::device_ptr<int64_t> vals(d_vertices);
  thrust::sort_by_key(keys, keys + n, vals);

  // for each partition p in 0, . . ., num_parts, finds first index where key >= p
  thrust::counting_iterator<int32_t> begin(0);
  thrust::device_vector<int64_t> offsets(num_parts + 1);

  thrust::lower_bound(keys, keys + n, begin, begin + num_parts, offsets.begin());

  offsets[num_partitions] = n;

  thrust::copy(offsets.begin(), offsets.end(), thrust::device_ptr<int64_t(d_partition_offsets));
}

__global__ void compute_edge_flags(int_64t n, int64_t* row_ptr, int32_t* col_idx, int32_t* partition_id, uint8_t* is_boundary){
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int32_t pid = partition_id[idx];
  // iterate over edges and flag them
  for (int_64_t i = row_ptr[idx]; i < row_ptr[idx + 1]; ++e){
    is_boundary[i] = (partition_id[col_idx[i]] != pid) ? 1 : 0;
  }
}

void get_subgraph(){
  
}

__device__ void compute_max_weight(){
  
}

__device__ float compute_goodness(int64_t u, int64_t v, float* max_incident_weight, float w_uv, float* M){
  float num = fmaxf(max_incident_weight[u], max_incident_weight[v]);
  float denom = fminf(fminf(M[u], M[v], w_uv);
  return num / denom;
}

__global__ void subgraphHAC(
			    int64_t* partition_vertices,
			    int64_t* row_ptr, int32_t* col_idx, float* weights,
			    int64_t split_offset, uint8_t* is_boundary,
			    float sim_threshold
			    int64_t* cluster_id
			    MergeEvent* merges,
			    int64_t* num_merges){
  
}
