// delta_stepping.cpp
//
// Basic-competent 1D distributed delta-stepping. See header for design notes.
// Optimizations deliberately omitted (relative to Chakaravarthy / Kaluś):
//   - Pruning (push/pull selection)         — push only
//   - Inner-Outer Short (IOS) heuristic
//   - Hybridization with Bellman-Ford
//   - Dynamic 16-bit local indices
//   - Vertex splitting for high-degree hubs
// Keep these in mind for the 2D comparison: adding any of them to the 1D
// baseline without an equivalent in 2D would bias the comparison.

#include "sssp_1d.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <stdexcept>

DeltaStepping::DeltaStepping(const CSRGraph& graph_,
                             int64_t source_,
                             double delta_,
                             MPI_Comm comm_)
    : graph(graph_), source(source_), delta(delta_), comm(comm_)
{
    if (delta <= 0.0) {
        throw std::invalid_argument("delta must be positive");
    }
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    dist.assign(graph.n_local, INF);
    prev_dist.assign(graph.n_local, INF);

    // Seed: source distance is 0 on whoever owns it.
    if (source >= graph.vertex_start && source < graph.vertex_end) {
        dist[source - graph.vertex_start] = 0.0;
    }

    // Initially every local vertex sits in next_buckets.
    next_buckets.resize(graph.n_local);
    std::iota(next_buckets.begin(), next_buckets.end(), int64_t{0});
}

bool DeltaStepping::find_next_bucket() {
    // Local min distance among not-yet-bucketed vertices.
    double local_min = INF;
    for (int64_t u : next_buckets) {
        if (dist[u] < local_min) local_min = dist[u];
    }

    double global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (global_min == INF) return false;  // all done

    const int64_t bucket_idx = static_cast<int64_t>(global_min / delta);
    const double  bucket_max = (bucket_idx + 1) * delta;  // exclusive

    // Move vertices with dist < bucket_max from next_buckets -> current_bucket.
    auto it = std::partition(
        next_buckets.begin(), next_buckets.end(),
        [&](int64_t u) { return dist[u] >= bucket_max; });

    current_bucket.assign(
        std::make_move_iterator(it),
        std::make_move_iterator(next_buckets.end()));
    next_buckets.erase(it, next_buckets.end());

    current_bucket_idx = bucket_idx;
    return true;
}

void DeltaStepping::process_bucket() {
    std::vector<int64_t> active = current_bucket;

    // Short-edge phases: iterate until no rank has any newly-active vertices.
    while (any_active_global(!active.empty())) {
        // Snapshot for change detection.
        std::copy(dist.begin(), dist.end(), prev_dist.begin());

        push_relax(active, /*short_only=*/true);
        active = recompute_active();
    }

    // Single long-edge phase: relax all w >= delta edges out of the
    // (now-settled) current_bucket. Targets land in strictly-later buckets
    // by construction (d_u + w >= k*delta + delta = (k+1)*delta), so they
    // can't bounce back into this bucket.
    push_relax(current_bucket, /*short_only=*/false);

    current_bucket.clear();
}

void DeltaStepping::push_relax(const std::vector<int64_t>& active,
                               bool short_only) {
    // Per-destination aggregation: at most one (target_local, min_proposed)
    // entry per (rank, target) pair. This keeps Alltoallv buffers small even
    // when many active vertices propose to the same target.
    std::vector<std::unordered_map<int64_t, double>> send_buf(p);

    for (int64_t u : active) {
        const double d_u = dist[u];
        if (d_u == INF) continue;  // shouldn't happen, but defensive

        const int64_t off_lo = graph.offsets[u];
        const int64_t off_hi = graph.offsets[u + 1];

        for (int64_t i = off_lo; i < off_hi; ++i) {
            const double w = graph.weighted ? graph.weights[i] : 1.0;

            // Short/long filter. Short = w < delta; long = w >= delta.
            if (short_only) { if (w >= delta) continue; }
            else            { if (w <  delta) continue; }

            const int64_t v_global = graph.neighbors[i];
            const double  proposed = d_u + w;

            const int     tgt_rank = static_cast<int>(
                owner_of(v_global, graph.n_global, p));
            const int64_t v_local  = global_to_local(v_global, graph.n_global, p);

            auto& slot = send_buf[tgt_rank];
            auto it = slot.find(v_local);
            if (it == slot.end()) {
                slot.emplace(v_local, proposed);
            } else if (proposed < it->second) {
                it->second = proposed;
            }
        }
    }

    // Flatten per-rank maps into contiguous send buffers.
    std::vector<int> send_counts(p, 0);
    int total_send = 0;
    for (int r = 0; r < p; ++r) {
        send_counts[r] = static_cast<int>(send_buf[r].size());
        total_send += send_counts[r];
    }
    std::vector<int> send_displs(p, 0);
    for (int r = 1; r < p; ++r) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    }

    std::vector<int64_t> send_targets(total_send);
    std::vector<double>  send_dists(total_send);
    for (int r = 0; r < p; ++r) {
        int idx = send_displs[r];
        for (const auto& kv : send_buf[r]) {
            send_targets[idx] = kv.first;
            send_dists[idx]   = kv.second;
            ++idx;
        }
    }

    // Exchange counts → exchange payloads.
    std::vector<int> recv_counts(p);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> recv_displs(p, 0);
    int total_recv = 0;
    for (int r = 0; r < p; ++r) {
        recv_displs[r] = total_recv;
        total_recv += recv_counts[r];
    }

    std::vector<int64_t> recv_targets(total_recv);
    std::vector<double>  recv_dists(total_recv);

    MPI_Alltoallv(send_targets.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T,
                  recv_targets.data(), recv_counts.data(), recv_displs.data(),
                  MPI_INT64_T, comm);

    MPI_Alltoallv(send_dists.data(), send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_dists.data(), recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE, comm);

    // Apply incoming relaxations locally.
    for (int i = 0; i < total_recv; ++i) {
        const int64_t v_local  = recv_targets[i];
        const double  proposed = recv_dists[i];
        if (proposed < dist[v_local]) {
            dist[v_local] = proposed;
        }
    }
}

std::vector<int64_t> DeltaStepping::recompute_active() {
    std::vector<int64_t> active;

    // (a) Vertices currently in the bucket whose distance changed.
    for (int64_t u : current_bucket) {
        if (dist[u] < prev_dist[u]) {
            active.push_back(u);
        }
    }

    // (b) Vertices from next_buckets that have now fallen into the current
    // bucket. Splice them into current_bucket and mark them active.
    const double bucket_max = (current_bucket_idx + 1) * delta;
    auto it = std::partition(
        next_buckets.begin(), next_buckets.end(),
        [&](int64_t u) { return dist[u] >= bucket_max; });

    for (auto jt = it; jt != next_buckets.end(); ++jt) {
        current_bucket.push_back(*jt);
        active.push_back(*jt);
    }
    next_buckets.erase(it, next_buckets.end());

    return active;
}

bool DeltaStepping::any_active_global(bool local_active) const {
    int local  = local_active ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LOR, comm);
    return global != 0;
}

void DeltaStepping::run() {
    while (find_next_bucket()) {
        process_bucket();
    }
}