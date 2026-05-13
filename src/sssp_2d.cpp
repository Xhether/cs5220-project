// sssp_2d.cpp
//
// 2D delta-stepping SSSP. See header for design notes.
//
// Bucket / short-phase / long-phase control flow is copied from DeltaStepping
// (sssp_1d). Only push_relax differs: instead of a single Alltoallv that
// routes proposals directly to vertex owners, the 2D version uses the
// BFS-2D pipeline:
//
//   1. TransposeVector  — pairwise (pr,pc) ↔ (pc,pr) swap, so active sources
//                         line up with the col_band of their new home.
//   2. col_comm Allgather — every rank in proc-col p_c collects all active
//                         sources in col_band[p_c].
//   3. Local CSC relax  — for each active source v in col_band, walk its
//                         in-tile predecessors u in row_band and compute
//                         d_v + w(u,v); locally aggregate min per u.
//   4. row_comm Alltoallv — bucket proposals by destination pc inside the
//                         proc-row, then exchange.
//   5. Apply locally.
//
// Same optimizations as 1D are deliberately omitted (pull, IOS, hybridization,
// vertex splitting, etc.).

#include "sssp_2d.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

DeltaStepping2D::DeltaStepping2D(const CSRGraph2D& graph_,
                                 int64_t source_,
                                 double delta_,
                                 MPI_Comm comm_)
    : graph(graph_), source(source_), delta(delta_), comm(comm_)
{
    if (delta <= 0.0) {
        throw std::invalid_argument("delta must be positive");
    }
    if (graph.grid_rows != graph.grid_cols) {
        throw std::runtime_error(
            "DeltaStepping2D requires a square processor grid (grid_rows == grid_cols)");
    }

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    R  = graph.grid_rows;
    pr = graph.pr;
    pc = graph.pc;
    partner_rank = pc * R + pr;

    // Keys make a rank's index inside the sub-comm equal its grid coordinate
    // (pc inside row_comm, pr inside col_comm). Same convention as bfs_2d.
    MPI_Comm_split(comm, pr, pc, &row_comm);
    MPI_Comm_split(comm, pc, pr, &col_comm);

    row_band_size = graph.row_end - graph.row_start;
    col_band_size = graph.col_end - graph.col_start;

    // Vector slice: row_band[pr] split among the R proc-cols (matches
    // bfs_2d_vec_range exactly).
    vec_start_  = graph.row_start + ((int64_t)pc * row_band_size) / R;
    vec_end_    = graph.row_start + ((int64_t)(pc + 1) * row_band_size) / R;
    n_vec_local = vec_end_ - vec_start_;

    dist.assign(n_vec_local, INF);
    prev_dist.assign(n_vec_local, INF);

    if (source >= vec_start_ && source < vec_end_) {
        dist[source - vec_start_] = 0.0;
    }

    next_buckets.resize(n_vec_local);
    std::iota(next_buckets.begin(), next_buckets.end(), int64_t{0});

    best.assign(graph.n_local_rows, INF);
    touched.reserve(graph.n_local_rows);

    build_csc();
}

DeltaStepping2D::~DeltaStepping2D() {
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

void DeltaStepping2D::build_csc() {
    csc_offsets.assign(col_band_size + 1, 0);
    for (int64_t k = 0; k < graph.m_local; k++) {
        int64_t v_local = graph.neighbors[k] - graph.col_start;
        csc_offsets[v_local + 1]++;
    }
    for (int64_t i = 1; i <= col_band_size; i++) {
        csc_offsets[i] += csc_offsets[i - 1];
    }

    csc_rows.resize(graph.m_local);
    if (graph.weighted) csc_weights.resize(graph.m_local);

    std::vector<int64_t> cursor = csc_offsets;
    for (int64_t u_local = 0; u_local < graph.n_local_rows; u_local++) {
        for (int64_t k = graph.offsets[u_local]; k < graph.offsets[u_local + 1]; k++) {
            int64_t v_local = graph.neighbors[k] - graph.col_start;
            int64_t dst = cursor[v_local]++;
            csc_rows[dst] = u_local;
            if (graph.weighted) csc_weights[dst] = graph.weights[k];
        }
    }
}

bool DeltaStepping2D::find_next_bucket() {
    double local_min = INF;
    for (int64_t u : next_buckets) {
        if (dist[u] < local_min) local_min = dist[u];
    }

    double global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (global_min == INF) return false;

    const int64_t bucket_idx = static_cast<int64_t>(global_min / delta);
    const double  bucket_max = (bucket_idx + 1) * delta;  // exclusive

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

void DeltaStepping2D::process_bucket() {
    std::vector<int64_t> active = current_bucket;

    while (any_active_global(!active.empty())) {
        std::copy(dist.begin(), dist.end(), prev_dist.begin());
        push_relax(active, /*short_only=*/true);
        active = recompute_active();
    }

    // Final long-edge phase: relax w >= delta out of the settled bucket. As in
    // 1D, targets land in strictly-later buckets so they can't bounce back.
    push_relax(current_bucket, /*short_only=*/false);

    current_bucket.clear();
}

void DeltaStepping2D::push_relax(const std::vector<int64_t>& active_local,
                                 bool short_only) {
    // ── Step 1: TransposeVector ──────────────────────────────────────────────
    // Pack our active sources as (global id, current distance), then pairwise
    // swap with rank (pc, pr). After the swap, the vector entries we hold are
    // for vertices in col_band[pc] (== row_band[pc], square grid) of our
    // new "column perspective".
    std::vector<int64_t> src_ids;
    std::vector<double>  src_dists;
    src_ids.reserve(active_local.size());
    src_dists.reserve(active_local.size());
    for (int64_t u_local : active_local) {
        src_ids.push_back(vec_start_ + u_local);
        src_dists.push_back(dist[u_local]);
    }

    std::vector<int64_t> t_ids;
    std::vector<double>  t_dists;
    if (partner_rank == rank) {
        t_ids   = std::move(src_ids);
        t_dists = std::move(src_dists);
    } else {
        int local_size   = static_cast<int>(src_ids.size());
        int partner_size = 0;
        MPI_Sendrecv(&local_size,   1, MPI_INT, partner_rank, 0,
                     &partner_size, 1, MPI_INT, partner_rank, 0,
                     comm, MPI_STATUS_IGNORE);
        t_ids.resize(partner_size);
        t_dists.resize(partner_size);
        MPI_Sendrecv(src_ids.data(),   local_size,   MPI_INT64_T, partner_rank, 1,
                     t_ids.data(),     partner_size, MPI_INT64_T, partner_rank, 1,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(src_dists.data(), local_size,   MPI_DOUBLE,  partner_rank, 2,
                     t_dists.data(),   partner_size, MPI_DOUBLE,  partner_rank, 2,
                     comm, MPI_STATUS_IGNORE);
    }

    // ── Step 2: col_comm Allgather (ids + dists in parallel) ─────────────────
    int local_count = static_cast<int>(t_ids.size());
    std::vector<int> ag_counts(R), ag_displs(R);
    MPI_Allgather(&local_count, 1, MPI_INT,
                  ag_counts.data(), 1, MPI_INT, col_comm);
    int total_ag = 0;
    for (int i = 0; i < R; i++) {
        ag_displs[i] = total_ag;
        total_ag += ag_counts[i];
    }

    std::vector<int64_t> fi_ids(total_ag);
    std::vector<double>  fi_dists(total_ag);
    MPI_Allgatherv(t_ids.data(),   local_count, MPI_INT64_T,
                   fi_ids.data(),  ag_counts.data(), ag_displs.data(),
                   MPI_INT64_T, col_comm);
    MPI_Allgatherv(t_dists.data(), local_count, MPI_DOUBLE,
                   fi_dists.data(), ag_counts.data(), ag_displs.data(),
                   MPI_DOUBLE, col_comm);

    // ── Step 3: Local CSC relax ──────────────────────────────────────────────
    // For each (v_global, d_v) in fi, walk csc_rows[v_local] in our row band
    // and aggregate min proposed distance per local row. `best` is kept INF
    // outside this section; `touched` records the rows we wrote so reset is
    // O(touched) rather than O(n_local_rows).
    touched.clear();

    for (int i = 0; i < total_ag; i++) {
        int64_t v_global = fi_ids[i];
        // Active sources after col-allgather should all live in our col band
        // (which equals row_band[pc]). Defensive bounds check.
        if (v_global < graph.col_start || v_global >= graph.col_end) continue;
        double d_v = fi_dists[i];
        if (d_v == INF) continue;

        int64_t v_local = v_global - graph.col_start;
        int64_t beg = csc_offsets[v_local];
        int64_t end = csc_offsets[v_local + 1];
        for (int64_t k = beg; k < end; k++) {
            double w = graph.weighted ? csc_weights[k] : 1.0;
            if (short_only) { if (w >= delta) continue; }
            else            { if (w <  delta) continue; }

            int64_t u_local  = csc_rows[k];
            double  proposed = d_v + w;
            double& slot     = best[u_local];
            if (proposed < slot) {
                if (slot == INF) touched.push_back(u_local);
                slot = proposed;
            }
        }
    }

    // ── Step 4: Bucket proposals by destination pc within row_comm ───────────
    std::vector<int> send_counts(R, 0);
    std::vector<int> send_displs(R, 0);
    std::vector<int> dest_pc(touched.size());

    for (size_t i = 0; i < touched.size(); i++) {
        int j = owner_pc_in_row_band(touched[i]);
        dest_pc[i] = j;
        send_counts[j]++;
    }
    int total_send = 0;
    for (int j = 0; j < R; j++) {
        send_displs[j] = total_send;
        total_send += send_counts[j];
    }

    std::vector<int64_t> send_ids(total_send);
    std::vector<double>  send_ds(total_send);
    std::vector<int>     cursor = send_displs;
    for (size_t i = 0; i < touched.size(); i++) {
        int64_t u_local = touched[i];
        int idx = cursor[dest_pc[i]]++;
        send_ids[idx] = graph.row_start + u_local;
        send_ds[idx]  = best[u_local];
    }
    // Reset scratch for next call.
    for (int64_t u_local : touched) best[u_local] = INF;

    // ── Step 5: row_comm Alltoallv (counts → ids → dists) ────────────────────
    std::vector<int> recv_counts(R);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, row_comm);

    std::vector<int> recv_displs(R, 0);
    int total_recv = 0;
    for (int j = 0; j < R; j++) {
        recv_displs[j] = total_recv;
        total_recv += recv_counts[j];
    }

    std::vector<int64_t> recv_ids(total_recv);
    std::vector<double>  recv_ds(total_recv);
    MPI_Alltoallv(send_ids.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T,
                  recv_ids.data(), recv_counts.data(), recv_displs.data(),
                  MPI_INT64_T, row_comm);
    MPI_Alltoallv(send_ds.data(),  send_counts.data(), send_displs.data(),
                  MPI_DOUBLE,
                  recv_ds.data(),  recv_counts.data(), recv_displs.data(),
                  MPI_DOUBLE, row_comm);

    // ── Step 6: Apply incoming relaxations to our vec slice ──────────────────
    for (int i = 0; i < total_recv; i++) {
        int64_t u_global = recv_ids[i];
        int64_t u_local  = u_global - vec_start_;
        if (u_local < 0 || u_local >= n_vec_local) continue;  // routing guard
        double proposed = recv_ds[i];
        if (proposed < dist[u_local]) dist[u_local] = proposed;
    }
}

std::vector<int64_t> DeltaStepping2D::recompute_active() {
    std::vector<int64_t> active;

    // (a) Bucket members whose dist dropped during this short-edge phase.
    for (int64_t u : current_bucket) {
        if (dist[u] < prev_dist[u]) active.push_back(u);
    }

    // (b) Vertices in next_buckets that have now fallen into the current
    // bucket: splice them into current_bucket and mark them active.
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

bool DeltaStepping2D::any_active_global(bool local_active) const {
    int local  = local_active ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LOR, comm);
    return global != 0;
}

int DeltaStepping2D::owner_pc_in_row_band(int64_t u_local_in_row_band) const {
    // Same split as bfs_2d_vec_range / bfs_2d's row_buckets routing:
    //   vec_split[j] = j * row_band_size / R
    // pc that owns offset `off` is the largest j with vec_split[j] <= off.
    int64_t off = u_local_in_row_band;
    int j = static_cast<int>((off * (int64_t)R) / row_band_size);
    if (j < 0) j = 0;
    if (j >= R) j = R - 1;
    while (j + 1 < R && ((int64_t)(j + 1) * row_band_size) / R <= off) j++;
    return j;
}

void DeltaStepping2D::run() {
    while (find_next_bucket()) {
        process_bucket();
    }
}
