#pragma once

#include <cstdint>

constexpr uint8_t kInactive = 0;
constexpr uint8_t kBlue = 1;
constexpr uint8_t kRed = 2;

struct GraphCSR {
    int    n;         // num vertices
    int    m;         // num edges (directed, so 2x undirected count)
    int*   row_ptr;   // length n+1
    int*   col_idx;   // length m
    float* weights;   // length m
};

struct ClusterState {
    int*     parent;        // length n, parent[i]==i means i is a root
    int*     cluster_size;  // length n, valid at roots only
    uint8_t* color;         // length n, kInactive/kBlue/kRed
    int*     merge_target;  // length n, scratch per inner round, -1 if none
    bool*    active;        // length n, true if i is a root still in play
};
