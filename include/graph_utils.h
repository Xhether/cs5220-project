#ifndef GRAPH_H
#define GRAPH_H

#include <cstdint>
#include <vector>
#include <string>
#include <mpi.h>

// Distributed CSR graph: each rank owns a contiguous block of vertices and their outgoing edges.
// Neighbor IDs are global vertex IDs.
// Also used to hold the full graph on rank 0 after serial loading, in which case
// vertex_start=0, vertex_end=n_local=n_global.
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

// 2D checkerboard-distributed CSR. Ranks form an R x C grid. A rank at grid
// position (pr, pc) owns the submatrix A[row_start:row_end, col_start:col_end]
// of the adjacency matrix: edges (u, v) with u in its row band and v in its
// column band. Locally stored as CSR over the row band, where each row holds
// only neighbors that fall in the column band.
struct CSRGraph2D {
    int64_t n_global;
    int64_t m_global;
    int     grid_rows;           // R
    int     grid_cols;           // C
    int     pr;                  // this rank's row in the grid
    int     pc;                  // this rank's column in the grid
    int64_t row_start, row_end;  // vertex rows owned (u range)
    int64_t col_start, col_end;  // vertex cols owned (v range)
    int64_t n_local_rows;        // row_end - row_start
    int64_t m_local;             // edges in this tile
    std::vector<int64_t> offsets;    // size n_local_rows + 1
    std::vector<int64_t> neighbors;  // global v IDs in [col_start, col_end)
    std::vector<double>  weights;
    bool weighted;
};

// Rank 0 reads a SNAP edge list, symmetrizes it, remaps vertices to a contiguous
// 0..n_global-1 range, and returns the full graph. Non-root ranks may call this
// too but will receive an empty CSRGraph (they should not); in practice only
// rank 0 should invoke it. No MPI communication is performed.
CSRGraph load_snap_graph_serial(const std::string& filename, bool assign_weights);

// Scatters a full graph held on rank 0 to a 1D block distribution across comm.
// On rank 0, `full` must be populated; on other ranks it is ignored.
CSRGraph distribute_graph_1d(const CSRGraph& full, MPI_Comm comm);

// Scatters a full graph held on rank 0 to a 2D grid of grid_rows x grid_cols
// ranks. Caller provides R and C with R*C == size(comm). Ranks are laid out
// in row-major order: rank = pr * grid_cols + pc.
CSRGraph2D distribute_graph_2d(const CSRGraph& full, int grid_rows, int grid_cols, MPI_Comm comm);

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
