#include "bfs_2d.h"
#include "mpi_utils.h"

#include <stdexcept>
#include <utility>
#include <vector>

void bfs_2d_vec_range(const CSRGraph2D& g, int64_t* vec_start, int64_t* vec_end) {
    int64_t row_band_size = g.row_end - g.row_start;
    *vec_start = g.row_start + ((int64_t)g.pc * row_band_size) / g.grid_cols;
    *vec_end   = g.row_start + ((int64_t)(g.pc + 1) * row_band_size) / g.grid_cols;
}

std::vector<int64_t> bfs_2d(const CSRGraph2D& g, int64_t source, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    if (g.grid_rows != g.grid_cols)
        throw std::runtime_error("bfs_2d: requires a square processor grid (grid_rows == grid_cols)");

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
    // csc_offsets[v_local + 1] − csc_offsets[v_local] = number of rows u_local
    // with edge (u_local, v_local) in this tile. csc_rows holds those u_local's.
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

    // ── Initial state ────────────────────────────────────────────────────────
    std::vector<int64_t> parents(n_vec_local, -1);
    std::vector<int64_t> frontier;
    if (source >= vec_start && source < vec_end) {
        parents[source - vec_start] = source;
        frontier.push_back(source);
    }

    // Transpose partner: pairwise swap (pr,pc) ↔ (pc,pr). Diagonal ranks are self-partners.
    const int partner_rank = pc * R + pr;

    while (true) {
        // ── Termination: any frontier non-empty? ─────────────────────────────
        int64_t local_fsize  = (int64_t)frontier.size();
        int64_t global_fsize = 0;
        MPI_Allreduce(&local_fsize, &global_fsize, 1, MPI_INT64_T, MPI_SUM, comm);
        if (global_fsize == 0) break;

        // ── Step 1: TransposeVector — pairwise swap (i,j) ↔ (j,i) ────────────
        std::vector<int64_t> f_transposed;
        if (partner_rank == rank) {
            f_transposed = frontier;
        } else {
            int local_size = (int)frontier.size();
            int partner_size = 0;
            MPI_Sendrecv(&local_size,   1, MPI_INT, partner_rank, 0,
                         &partner_size, 1, MPI_INT, partner_rank, 0,
                         comm, MPI_STATUS_IGNORE);
            f_transposed.resize(partner_size);
            MPI_Sendrecv(frontier.data(),     local_size,   MPI_INT64_T, partner_rank, 1,
                         f_transposed.data(), partner_size, MPI_INT64_T, partner_rank, 1,
                         comm, MPI_STATUS_IGNORE);
        }

        // ── Step 2: Allgatherv on col_comm → fi (full col band j) ────────────
        int local_count = (int)f_transposed.size();
        std::vector<int> ag_counts(R), ag_displs(R);
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

        // ── Step 3: Local SpMSV (column-driven, sparse accumulator) ──────────
        // For each frontier vertex v, walk its column in the local CSC and
        // claim each row u as discovered with parent v (first claim wins).
        std::vector<int64_t> t_parents(g.n_local_rows, -1);
        for (int64_t v : fi) {
            int64_t v_local = v - g.col_start;
            int64_t beg = csc_offsets[v_local];
            int64_t end = csc_offsets[v_local + 1];
            for (int64_t k = beg; k < end; k++) {
                int64_t u_local = csc_rows[k];
                if (t_parents[u_local] == -1)
                    t_parents[u_local] = v;
            }
        }

        // ── Step 4: Alltoallv along row_comm — send (u, parent) to u's owner ─
        // Each pair is two consecutive int64s in the bucket; the underlying
        // SendBuffer counts elements (so 2 per pair).
        std::vector<std::vector<int64_t>> row_buckets(R);
        for (int64_t u_local = 0; u_local < g.n_local_rows; u_local++) {
            int64_t parent = t_parents[u_local];
            if (parent == -1) continue;
            int64_t u = g.row_start + u_local;
            // Invert the pc-partition (pc*size)/R with an adjust loop — a plain
            // (offset*R)/size undershoots on exact boundary vertices. Mirrors
            // graph_utils.cpp:owner_of.
            int64_t off = u - g.row_start;
            int j_dest = (int)((off * (int64_t)R) / row_band_size);
            if (j_dest < 0) j_dest = 0;
            if (j_dest >= R) j_dest = R - 1;
            while (j_dest + 1 < R && ((int64_t)(j_dest + 1) * row_band_size) / R <= off)
                j_dest++;
            row_buckets[j_dest].push_back(u);
            row_buckets[j_dest].push_back(parent);
        }
        std::vector<int64_t> recv_pairs = mpi_utils::alltoallv_exchange(row_buckets, row_comm);

        // ── Step 5: Mask + update ────────────────────────────────────────────
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

        frontier = std::move(new_frontier);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    return parents;
}
