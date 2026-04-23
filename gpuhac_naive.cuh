#include "cuda.h"
#include <cassert>
#include "gpuhac_naive_types.h"

__device__ int find(int* parent, int v) {
  while (parent[v] != v) {
    parent[v] = parent[parent[v]]; 
    v = parent[v];
  }
  return v;
}


/*__device__ void union_clusters(int* parent, int* cluster_size, int u, int v) {
  assert(parent[u] == u && parent[v] == v); // need these to be roots
  
  if (cluster_size[u] < cluster_size[v]) {
    int tmp = u; u = v; v = tmp;
  }
  // v merges into u
  atomicExch(&parent[v], u);
  atomicAdd(&cluster_size[u], cluster_size[v]);
}
*/

// helper for compute_max_weight_kernel to work w/ floats
// only works for positive floats. fine for weights?
__device__ void atomicMaxFloat(float* addr, float val) {
  atomicMax((int*)addr, __float_as_int(val));
}

__global__ void compute_max_weight_kernel(GraphCSR G, ClusterState CS, float* out_max_weight)  {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= G.n) return;

  int root_i = find(CS.parent, i);
  float local_max = 0.0f;

  for (int e = G.row_ptr[i]; e < G.row_ptr[i+1]; e++) {
    int j = G.col_idx[e];
    int root_j = find(CS.parent, j);
    if (root_i == root_j) continue;

    float w = G.weights[e] / (CS.cluster_size[root_i] * CS.cluster_size[root_j]);
    if (w > local_max) local_max = w;
  }

  if (local_max > 0.0f) atomicMaxFloat(out_max_weight, local_max);
}

// integer scramble helper for nearly-random coloring (might change to something else)
__device__ unsigned int hash_uint(unsigned int x) {
  x ^= x >> 16;
  x *= 0x45d9f3b;
  x ^= x >> 16;
  return x;
}

__global__ void color_vertices_kernel(ClusterState CS, unsigned long long seed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!CS.active[i]) return;

  unsigned int h = hash_uint((unsigned int)(seed ^ i));
  CS.color[i] = (h & 1) ? kRed : kBlue;
}

// out_red should be init to INT_MAX -> INT_MAX values means no red neighbor found
__global__ void build_triples_kernel(GraphCSR G,
				     ClusterState CS,
				     int* out_red,
				     float* out_priority,
				     unsigned long long seed,
				     int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= G.n) return;
  
  int root_i = find(CS.parent, i);
  if (CS.color[root_i] != kBlue) return;
  
  for (int e = G.row_ptr[i]; e < G.row_ptr[i+1]; e++) {
    int j = G.col_idx[e];
    int root_j = find(CS.parent, j);
    if (root_j == root_i) continue; // same cluster
    if (!CS.active[root_j]) continue;
    if (CS.color[root_j] != kRed) continue;
    
    // maybe a race. handles deterministically
    atomicMin(&out_red[root_i], root_j);
    
    // benign write race
    out_priority[root_i] =
      (float)hash_uint((unsigned int)(seed ^ (unsigned)root_i ^ 0xdeadbeef))
      / (float)UINT_MAX;
    
    break; // this thread found a red; potential race handled by atomicMin
  }
}


__global__ void prefix_select_kernel(ClusterState CS,
				     int* sorted_blue,
				     int* red_offsets,
				     int* red_roots,
				     float one_plus_eps,
				     int num_red,
				     int* out_merge_target) // length n
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_red) return;
  
  int r       = red_roots[idx];
  int start   = red_offsets[idx];
  int end     = red_offsets[idx + 1];
  int r_size  = CS.cluster_size[r];
  int accumulated = 0;
  
  for (int i = start; i < end; i++) {
    int b      = sorted_blue[i];
    int b_size = CS.cluster_size[b];
    if (accumulated + b_size <= (int)(one_plus_eps * r_size)) {
      out_merge_target[b] = r;
      accumulated += b_size;
    } else {
      break;
    }
  }
}

// directed union of clusters, dst stays root, src merges
__device__ void union_into(int* parent, int* cluster_size, int src, int dst) {
    assert(parent[src] == src && parent[dst] == dst);
    atomicExch(&parent[src], dst);
    atomicAdd(&cluster_size[dst], cluster_size[src]);
}

__global__ void apply_merges_kernel(ClusterState CS,
				    int* merge_target,  // length n, blue -> red root, -1 if no merge
				    int  n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!CS.active[i]) return;
  if (CS.color[i] != kBlue) return;
  
  int r = merge_target[i];
  if (r == -1) return;
  
  union_into(CS.parent, CS.cluster_size, i, r);

  // blue merged into red, mark blue as inactive
  CS.active[i] = false;
}
