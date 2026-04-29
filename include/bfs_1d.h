#ifndef BFS_1D_H
#define BFS_1D_H

#include "bfs_timing.h"
#include "graph_utils.h"
#include <cstdint>
#include <vector>

// Parallel BFS from `source` on a 1D row-partitioned graph.
// Fills d (size g.n_local) with hop counts from source; unreachable vertices
// get INF (std::numeric_limits<int64_t>::max()). Collective over MPI_COMM_WORLD.
// If `timing` is non-null, fills it with the slowest-rank wall times for the
// BFS traversal (gather/output, if any, are excluded).
void bfs_1d(const CSRGraph& g, int64_t source, std::vector<int64_t>& d,
            BFSTiming* timing = nullptr);

#endif // BFS_1D_H
