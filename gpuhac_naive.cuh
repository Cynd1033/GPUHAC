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


__device__ void union_clusters(int* parent, int* cluster_size, int u, int v) {
  assert(parent[u] == u && parent[v] == v); // need these to be roots
  
  if (cluster_size[u] < cluster_size[v]) {
    int tmp = u; u = v; v = tmp;
  }
  // v merges into u
  atomicExch(&parent[v], u);
  atomicAdd(&cluster_size[u], cluster_size[v]);
}


// helper for compute_max_weight_kernel to work w/ floats
__device__ void atomicMaxFloat(float* addr, float val) {
  atomicMax((int*)addr, __float_as_int(val));
}

__global__ void compute_max_weight_kernel(GraphCSR G, ClusterState CS, float* out_max_weight)  {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= G.n) return;
  if (!CS.active[i]) return;

  int root_i = find(CS.parent, i);
  float local_max = 0.0f;

  for (int e = G.row_ptr[i]; e < G.row_ptr[i+1]; e++) {
    int j = G.col_idx[e];
    int root_j = find(CS.parent, j);
    if (root_i == root_j) continue;

    float w = G.weights[e] / (CS.cluster_size[root_i] * CS.cluster_size[root_j]);
    if (w > local_max) local_max = w;
  }

  atomicMaxFloat(out_max_weight, local_max);
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

__global__ void build_triples_kernel(GraphCSR G,
				     ClusterState CS,
				     int* out_red,
				     float* out_priority,
				     float* out_weight, // for tracking weight through merges
				     unsigned long long seed,
				     int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!CS.active[i]) return;
  if (CS.color[i] != kBlue) return;

  int num_red = 0;
  for (int e = G.row_ptr[i]; e < G.row_ptr[i+1]; e++) {
    int j = G.col_idx[e];
    int root_j = find(CS.parent, j);
    if (root_j == i) continue;  
    if (!CS.active[root_j]) continue;
    if (CS.color[root_j] == kRed) num_red++;
  }

  if (num_red == 0) {
    out_red[i] = -1;
    return;
  }

  unsigned int h1 = hash_uint((unsigned int)(seed ^ i));
  int chosen_idx = h1 % num_red;

  // this double pass is terrible
  // TODO: fix w/ resevoir sampling
  int cur = 0;
  for (int e = G.row_ptr[i]; e < G.row_ptr[i+1]; e++) {
    int j = G.col_idx[e];
    int root_j = find(CS.parent, j);
    if (root_j == i) continue;
    if (!CS.active[root_j]) continue;
    if (CS.color[root_j] == kRed) {
      if (cur == chosen_idx) {
	out_red[i] = root_j;
	out_weight[i] = G.weights[e] / (CS.cluster_size[i] * CS.cluster_size[root_j]);
	break;
      }
      cur++;
    }
  }

  unsigned int h2 = hash_uint((unsigned int)(seed ^ i ^ 0xdeadbeef));
  out_priority[i] = (float)h2 / (float)UINT_MAX;
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
  
  // both should be roots
  assert(CS.parent[i] == i);
  assert(CS.parent[r] == r);
  
  union_clusters(CS.parent, CS.cluster_size, i, r);

  // blue merged into red, mark blue as inactive
  CS.active[i] = false;
}
