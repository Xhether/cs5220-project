#include "bfs_2d.h"
#include "mpi_utils.h"

#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

void bfs_2d_vec_range(const CSRGraph2D& g, int64_t* vec_start, int64_t* vec_end) {
    int64_t row_band_size = g.row_end - g.row_start;
    *vec_start = g.row_start + ((int64_t)g.pc * row_band_size) / g.grid_cols;
    *vec_end   = g.row_start + ((int64_t)(g.pc + 1) * row_band_size) / g.grid_cols;
}

std::vector<int64_t> bfs_2d(const CSRGraph2D& g, int64_t source, MPI_Comm comm,
                            BFSTiming* timing) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    if (g.grid_rows != g.grid_cols)
        throw std::runtime_error("bfs_2d: requires a square processor grid (grid_rows == grid_cols)");

    double t_total_start = MPI_Wtime();
    double t_comm = 0.0;

    // Per-phase accumulators
    double t_csc_build = 0.0;
    double t_termcheck = 0.0;
    double t_transpose = 0.0;
    double t_expand    = 0.0;
    double t_spmv      = 0.0;
    double t_fold      = 0.0;
    double t_mask      = 0.0;

    const int R = g.grid_rows;       // == grid_cols
    const int pr = g.pr;
    const int pc = g.pc;

    // Row & column communicators. Keys are chosen so that a rank's index inside
    // the sub-comm equals its grid coordinate (pc inside row_comm, pr inside col_comm).
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(comm, pr, pc, &row_comm);
    MPI_Comm_split(comm, pc, pr, &col_comm);

    // Local 2D vector slice
    int64_t vec_start, vec_end;
    bfs_2d_vec_range(g, &vec_start, &vec_end);
    const int64_t n_vec_local   = vec_end - vec_start;
    const int64_t row_band_size = g.row_end - g.row_start;
    const int64_t col_band_size = g.col_end - g.col_start;

    // ── Build local CSC of the tile ──────────────────────────────────────────
    double t0_csc = MPI_Wtime();
    std::vector<int64_t> csc_offsets(col_band_size + 1, 0);
    for (int64_t k = 0; k < g.m_local; k++) {
        int64_t v_local = g.neighbors[k] - g.col_start;
        csc_offsets[v_local + 1]++;
    }
    for (int64_t i = 1; i <= col_band_size; i++)
        csc_offsets[i] += csc_offsets[i - 1];

    std::vector<int64_t> csc_rows(g.m_local);
    {
        std::vector<int64_t> cursor = csc_offsets;
        for (int64_t u_local = 0; u_local < g.n_local_rows; u_local++) {
            for (int64_t k = g.offsets[u_local]; k < g.offsets[u_local + 1]; k++) {
                int64_t v_local = g.neighbors[k] - g.col_start;
                csc_rows[cursor[v_local]++] = u_local;
            }
        }
    }
    t_csc_build = MPI_Wtime() - t0_csc;

    // ── Initial state ────────────────────────────────────────────────────────
    std::vector<int64_t> parents(n_vec_local, -1);
    std::vector<int64_t> frontier;
    if (source >= vec_start && source < vec_end) {
        parents[source - vec_start] = source;
        frontier.push_back(source);
    }

    // Transpose partner: pairwise swap (pr,pc) ↔ (pc,pr). Diagonal ranks are self-partners.
    const int partner_rank = pc * R + pr;

    // Per-level scratch space, reused across iterations. t_parents is kept
    // dense for O(1) "first claim wins" checks during SpMSV, but only entries
    // recorded in t_discovered are touched, so reset cost is O(frontier size)
    // rather than O(n_local_rows) — matters for high-diameter graphs.
    std::vector<int64_t> t_parents(g.n_local_rows, -1);
    std::vector<int64_t> t_discovered;

    while (true) {
        // ── Termination: any frontier non-empty? ─────────────────────────────
        int64_t local_fsize  = (int64_t)frontier.size();
        int64_t global_fsize = 0;
        double tc1 = MPI_Wtime();
        MPI_Allreduce(&local_fsize, &global_fsize, 1, MPI_INT64_T, MPI_SUM, comm);
        double dt = MPI_Wtime() - tc1;
        t_comm      += dt;
        t_termcheck += dt;
        if (global_fsize == 0) break;

        // ── Step 1: TransposeVector — pairwise swap (i,j) ↔ (j,i) ────────────
        std::vector<int64_t> f_transposed;
        if (partner_rank == rank) {
            f_transposed = frontier;
        } else {
            int local_size = (int)frontier.size();
            int partner_size = 0;
            tc1 = MPI_Wtime();
            MPI_Sendrecv(&local_size,   1, MPI_INT, partner_rank, 0,
                         &partner_size, 1, MPI_INT, partner_rank, 0,
                         comm, MPI_STATUS_IGNORE);
            f_transposed.resize(partner_size);
            MPI_Sendrecv(frontier.data(),     local_size,   MPI_INT64_T, partner_rank, 1,
                         f_transposed.data(), partner_size, MPI_INT64_T, partner_rank, 1,
                         comm, MPI_STATUS_IGNORE);
            dt = MPI_Wtime() - tc1;
            t_comm      += dt;
            t_transpose += dt;
        }

        // ── Step 2: Allgather + Allgatherv on col_comm → fi ──────────────────
        int local_count = (int)f_transposed.size();
        std::vector<int> ag_counts(R), ag_displs(R);
        tc1 = MPI_Wtime();
        MPI_Allgather(&local_count, 1, MPI_INT,
                      ag_counts.data(), 1, MPI_INT, col_comm);
        int total_ag = 0;
        for (int i = 0; i < R; i++) {
            ag_displs[i] = total_ag;
            total_ag += ag_counts[i];
        }
        std::vector<int64_t> fi(total_ag);
        MPI_Allgatherv(f_transposed.data(), local_count, MPI_INT64_T,
                       fi.data(), ag_counts.data(), ag_displs.data(), MPI_INT64_T,
                       col_comm);
        dt = MPI_Wtime() - tc1;
        t_comm   += dt;
        t_expand += dt;

        // ── Step 3: Local SpMSV (column-driven, sparse accumulator) ──────────
        tc1 = MPI_Wtime();
        t_discovered.clear();
        for (int64_t v : fi) {
            int64_t v_local = v - g.col_start;
            int64_t beg = csc_offsets[v_local];
            int64_t end = csc_offsets[v_local + 1];
            for (int64_t k = beg; k < end; k++) {
                int64_t u_local = csc_rows[k];
                if (t_parents[u_local] == -1) {
                    t_parents[u_local] = v;
                    t_discovered.push_back(u_local);
                }
            }
        }
        t_spmv += MPI_Wtime() - tc1;

        // ── Step 4: Build row_buckets + alltoallv on row_comm ─────────────────
        tc1 = MPI_Wtime();
        std::vector<std::vector<int64_t>> row_buckets(R);
        for (int64_t u_local : t_discovered) {
            int64_t parent = t_parents[u_local];
            int64_t u = g.row_start + u_local;
            int64_t off = u - g.row_start;
            int j_dest = (int)((off * (int64_t)R) / row_band_size);
            if (j_dest < 0) j_dest = 0;
            if (j_dest >= R) j_dest = R - 1;
            while (j_dest + 1 < R && ((int64_t)(j_dest + 1) * row_band_size) / R <= off)
                j_dest++;
            row_buckets[j_dest].push_back(u);
            row_buckets[j_dest].push_back(parent);
        }
        // Sparse reset: only clear the entries we touched this level.
        for (int64_t u_local : t_discovered) t_parents[u_local] = -1;
        std::vector<int64_t> recv_pairs = mpi_utils::alltoallv_exchange(row_buckets, row_comm);
        dt = MPI_Wtime() - tc1;
        t_comm += dt;
        t_fold  += dt;

        // ── Step 5: Mask + update ────────────────────────────────────────────
        tc1 = MPI_Wtime();
        std::vector<int64_t> new_frontier;
        new_frontier.reserve(recv_pairs.size() / 2);
        for (size_t i = 0; i + 1 < recv_pairs.size(); i += 2) {
            int64_t u      = recv_pairs[i];
            int64_t parent = recv_pairs[i + 1];
            int64_t u_local = u - vec_start;
            if (u_local < 0 || u_local >= n_vec_local) continue;  // routing bug guard
            if (parents[u_local] == -1) {
                parents[u_local] = parent;
                new_frontier.push_back(u);
            }
        }
        t_mask += MPI_Wtime() - tc1;

        frontier = std::move(new_frontier);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    double t_total = MPI_Wtime() - t_total_start;

    // Reduce all timers to max across ranks
    double vals[8] = { t_total, t_comm, t_csc_build, t_termcheck,
                       t_transpose, t_expand, t_spmv, t_fold };
    double maxv[8] = {};
    MPI_Reduce(vals, maxv, 8, MPI_DOUBLE, MPI_MAX, 0, comm);
    double max_mask = 0;
    MPI_Reduce(&t_mask, &max_mask, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        double max_total   = maxv[0];
        double max_comm    = maxv[1];
        double max_csc     = maxv[2];
        double max_term    = maxv[3];
        double max_trans   = maxv[4];
        double max_expand  = maxv[5];
        double max_spmv    = maxv[6];
        double max_fold    = maxv[7];
        fprintf(stderr,
            "[bfs2d phases] csc_build=%.4f termcheck=%.4f transpose=%.4f"
            " expand=%.4f spmv=%.4f fold=%.4f mask=%.4f"
            " | total=%.4f comm=%.4f compute=%.4f\n",
            max_csc, max_term, max_trans,
            max_expand, max_spmv, max_fold, max_mask,
            max_total, max_comm, max_total - max_comm);

        if (timing) {
            timing->total_time   = max_total;
            timing->comm_time    = max_comm;
            timing->compute_time = max_total - max_comm;
        }
    }

    return parents;
}
