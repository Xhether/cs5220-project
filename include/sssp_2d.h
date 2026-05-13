// sssp_2d.h
//
// 2D distributed delta-stepping SSSP over a checkerboard-partitioned CSR graph
// (CSRGraph2D). Communication pipeline mirrors bfs_2d:
//     transpose → col-allgather → local CSC relax → row-alltoallv.
// Bucket/short/long control flow is the same as DeltaStepping (1D); only the
// push_relax step changes shape.
//
// Requires a square processor grid (grid_rows == grid_cols), so that the row-
// band of proc-row r exactly matches the col-band of proc-col r and the
// transpose-then-column-allgather pattern lines up.
//
// Usage:
//     CSRGraph2D g = distribute_graph_2d(full, R, R, MPI_COMM_WORLD);
//     DeltaStepping2D ds(g, /*source=*/0, /*delta=*/10.0, MPI_COMM_WORLD);
//     ds.run();
//     auto& d = ds.distances();   // slice for [vec_start(), vec_end())

#pragma once

#include <vector>
#include <limits>
#include <cstdint>
#include <mpi.h>

#include "graph_utils.h"

class DeltaStepping2D {
public:
    static constexpr double INF = std::numeric_limits<double>::infinity();

    DeltaStepping2D(const CSRGraph2D& graph,
                    int64_t source,
                    double delta,
                    MPI_Comm comm);

    ~DeltaStepping2D();

    DeltaStepping2D(const DeltaStepping2D&)            = delete;
    DeltaStepping2D& operator=(const DeltaStepping2D&) = delete;

    void run();

    // Local distances over the 2D vector slice [vec_start(), vec_end()).
    // distances()[i] is the shortest distance to global vertex vec_start()+i;
    // INF if unreachable.
    const std::vector<double>& distances() const { return dist; }

    int64_t vec_start() const { return vec_start_; }
    int64_t vec_end()   const { return vec_end_; }

private:
    const CSRGraph2D& graph;
    int64_t source;
    double  delta;
    MPI_Comm comm;
    MPI_Comm row_comm;   // ranks sharing pr; key=pc so index_in_comm == pc
    MPI_Comm col_comm;   // ranks sharing pc; key=pr so index_in_comm == pr
    int rank;
    int p;
    int R;               // grid_rows == grid_cols
    int pr;
    int pc;
    int partner_rank;    // transpose partner (pc * R + pr)

    // Vector partition: row_band[pr] further sliced among the R proc-cols.
    int64_t vec_start_;
    int64_t vec_end_;
    int64_t n_vec_local;
    int64_t row_band_size;
    int64_t col_band_size;

    // CSC of the local tile keyed on v_local ∈ [0, col_band_size). Each entry
    // gives the row (u_local ∈ [0, n_local_rows)) and the corresponding weight.
    std::vector<int64_t> csc_offsets;
    std::vector<int64_t> csc_rows;
    std::vector<double>  csc_weights;

    // dist over [0, n_vec_local). prev_dist is a snapshot used by short-edge
    // phases to detect which bucket members became (re)active.
    std::vector<double> dist;
    std::vector<double> prev_dist;

    std::vector<int64_t> next_buckets;
    std::vector<int64_t> current_bucket;
    int64_t current_bucket_idx = -1;

    // Scratch reused across push_relax calls — kept here to avoid per-phase
    // allocations. best[u_local]=INF outside of the inner aggregation.
    std::vector<double>  best;
    std::vector<int64_t> touched;

    void build_csc();
    bool find_next_bucket();
    void process_bucket();
    void push_relax(const std::vector<int64_t>& active_local, bool short_only);
    std::vector<int64_t> recompute_active();
    bool any_active_global(bool local_active) const;

    // Which proc-column in our row owns u_local (a vec-slice index within our
    // row band). Mirrors bfs_2d's row_buckets routing.
    int owner_pc_in_row_band(int64_t u_local_in_row_band) const;
};
