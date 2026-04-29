#include "bfs_1d.h"
#include "mpi_utils.h"

#include <mpi.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

void bfs_1d(const CSRGraph& g, int64_t source, std::vector<int64_t>& d,
            BFSTiming* timing) {
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    const int64_t INF = std::numeric_limits<int64_t>::max();

    double t_total_start = MPI_Wtime();
    double t_comm = 0.0;

    // Initialize distances; seed the source on its owning rank.
    d.assign(g.n_local, INF);
    std::vector<int64_t> frontier;
    int src_owner = owner_of(source, g.n_global, p);
    if (rank == src_owner) {
        int64_t src_local = global_to_local(source, g.n_global, p);
        d[src_local] = 0;
        frontier.push_back(src_local);
    }

    mpi_utils::FrontierPacker packer(p);
    int64_t level = 1;

    while (true) {
        // Step 1: termination check — global frontier size.
        int64_t local_fsize  = (int64_t)frontier.size();
        int64_t global_fsize = 0;
        double tc1 = MPI_Wtime();
        MPI_Allreduce(&local_fsize, &global_fsize, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - tc1;
        if (global_fsize == 0) break;

        // Step 2: expand frontier — route every neighbor to its owner.
        packer.clear();
        for (size_t i = 0; i < frontier.size(); i++) {
            int64_t u = frontier[i];
            int64_t deg = local_degree(g, u);
            const int64_t* nbrs = local_neighbors_ptr(g, u);
            for (int64_t k = 0; k < deg; k++) {
                int64_t v = nbrs[k];
                int owner = owner_of(v, g.n_global, p);
                packer.add(v, owner);
            }
        }

        // Step 3: exchange via alltoallv.
        tc1 = MPI_Wtime();
        std::vector<int64_t> received = mpi_utils::alltoallv_exchange(packer.pack(), MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - tc1;

        // Step 4: build next frontier — claim unvisited vertices at this level.
        std::vector<int64_t> next_frontier;
        next_frontier.reserve(received.size());
        for (size_t i = 0; i < received.size(); i++) {
            int64_t v_local = global_to_local(received[i], g.n_global, p);
            if (d[v_local] == INF) {
                d[v_local] = level;
                next_frontier.push_back(v_local);
            }
        }

        // Step 5: swap frontiers and advance level.
        frontier = std::move(next_frontier);
        level++;
    }

    double t_total = MPI_Wtime() - t_total_start;

    // Reduce per-rank times to the slowest (max) across the comm.
    double max_total = 0, max_comm = 0;
    MPI_Reduce(&t_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,  &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (timing && rank == 0) {
        timing->total_time   = max_total;
        timing->comm_time    = max_comm;
        timing->compute_time = max_total - max_comm;
    }
}
