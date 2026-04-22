#ifndef GRAPH_H
#define GRAPH_H

#include <cstdint>
#include <vector>
#include <string>
#include <mpi.h>

// Distributed CSR graph: each rank owns a contiguous block of vertices and their outgoing edges.
// Neighbor IDs are global vertex IDs
struct CSRGraph {
    int64_t n_global;            // total vertices across all ranks
    int64_t m_global;            // total edges (both directions for undirected)
    int64_t vertex_start;        // first global vertex ID owned by this rank
    int64_t vertex_end;          // exclusive upper bound of owned vertex IDs
    int64_t n_local;             // number of locally owned vertices
    int64_t m_local;             // number of edges out of locally owned vertices
    std::vector<int64_t> offsets;     // CSR row pointers, size n_local + 1
    std::vector<int64_t> neighbors;   // global neighbor IDs, size m_local
    std::vector<double>  weights;     // edge weights, size m_local; empty if unweighted
    bool weighted;
};

// Loads a SNAP edge list, symmetrizes it, and distributes it across ranks.
// If assign_weights is true, fills weights[] with deterministic values in [1, 100].
CSRGraph load_snap_graph(const std::string& filename, bool assign_weights, MPI_Comm comm);

// Returns the rank that owns the given global vertex ID (1D block distribution).
int owner_of(int64_t global_vertex, int64_t n_global, int p);

// Converts a global vertex ID to a local index within its owning rank's slice.
int64_t global_to_local(int64_t global_vertex, int64_t n_global, int p);

// Converts a local index on the given rank back to a global vertex ID.
int64_t local_to_global(int64_t local_index, int rank, int64_t n_global, int p);

// Returns a deterministic weight in [1, 100] for edge (u, v); symmetric in u and v.
double weight_of(int64_t u, int64_t v);

// Returns the degree of a locally owned vertex given its local index.
inline int64_t local_degree(const CSRGraph& g, int64_t local_v) {
    return g.offsets[local_v + 1] - g.offsets[local_v];
}

// Returns a pointer to the first neighbor of a locally owned vertex.
inline const int64_t* local_neighbors_ptr(const CSRGraph& g, int64_t local_v) {
    return &g.neighbors[g.offsets[local_v]];
}

// Returns a pointer to the first weight of a locally owned vertex, or nullptr if unweighted.
inline const double* local_weights_ptr(const CSRGraph& g, int64_t local_v) {
    return g.weighted ? &g.weights[g.offsets[local_v]] : nullptr;
}

// Prints per-rank stats: vertex range, edge count, and min/max/avg local degree.
void print_graph_stats(const CSRGraph& g, MPI_Comm comm);

#endif  