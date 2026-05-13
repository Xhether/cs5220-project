// delta_stepping.h
//
// 1D distributed delta-stepping SSSP over a row-block-partitioned CSR graph.
// Uses MPI bulk-synchronous communication (Alltoallv per phase) with local
// aggregation. Push-only — no pull/IOS/hybridization/vertex-splitting yet.
//
// Usage:
//     CSRGraph local = distribute_graph_1d(full, MPI_COMM_WORLD);
//     DeltaStepping ds(local, /*source=*/0, /*delta=*/10.0, MPI_COMM_WORLD);
//     ds.run();
//     auto& d = ds.distances();   // local slice: vertices [vertex_start, vertex_end)

#pragma once

#include <vector>
#include <limits>
#include <cstdint>
#include <mpi.h>

#include "graph_utils.h"

class DeltaStepping {
public:
    static constexpr double INF = std::numeric_limits<double>::infinity();

    DeltaStepping(const CSRGraph& graph,
                  int64_t source,
                  double delta,
                  MPI_Comm comm);

    // Run delta-stepping to completion.
    void run();

    // Local distances. distances()[i] is the shortest distance to global
    // vertex (graph.vertex_start + i). INF for unreachable vertices.
    const std::vector<double>& distances() const { return dist; }

private:
    const CSRGraph& graph;
    int64_t source;
    double  delta;
    MPI_Comm comm;
    int rank;
    int p;

    // Distance vector, indexed by local vertex index [0, graph.n_local).
    std::vector<double> dist;
    std::vector<double> prev_dist;  // snapshot at start of each short-edge phase

    // Vertices not yet placed into the current bucket. Each vertex sits here
    // until find_next_bucket() promotes it (or it gets dropped into the
    // current bucket mid-phase via recompute_active()).
    std::vector<int64_t> next_buckets;

    // Vertices currently being settled.
    std::vector<int64_t> current_bucket;
    int64_t current_bucket_idx = -1;

    // Find the globally-minimum next bucket and move its members from
    // next_buckets to current_bucket. Returns false iff no work remains.
    bool find_next_bucket();

    // Settle current_bucket: alternating short-edge phases until globally
    // empty, then one long-edge phase.
    void process_bucket();

    // Push relaxations from `active`. If short_only, only edges with w < delta;
    // otherwise only edges with w >= delta.
    void push_relax(const std::vector<int64_t>& active, bool short_only);

    // After a short-edge phase: build the next active set as
    //   (a) vertices in current_bucket whose dist decreased, plus
    //   (b) vertices in next_buckets that have now fallen into current bucket
    //       (these also get spliced into current_bucket).
    std::vector<int64_t> recompute_active();

    // Allreduce-OR for "any rank still has work in this bucket."
    bool any_active_global(bool local_active) const;
};