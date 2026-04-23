#ifndef BFS_2D_H
#define BFS_2D_H

#include "graph_utils.h"

#include <cstdint>
#include <mpi.h>
#include <vector>

// Vertex slice owned by this rank in the 2D vector distribution. The vector is
// partitioned into grid_rows row-bands (matching the matrix), and each row-band
// is then split into grid_cols pieces among the ranks of that processor row.
void bfs_2d_vec_range(const CSRGraph2D& g, int64_t* vec_start, int64_t* vec_end);

// Parallel BFS from `source` on a 2D-distributed graph (Buluç & Madduri SC'11).
// Returns the local slice of the parents array, indexed by (v - vec_start);
// parents[i] is the predecessor of vertex (vec_start + i), or -1 if unreachable.
// The source's slot is set to `source` itself.
//
// Requires grid_rows == grid_cols (square processor grid).
std::vector<int64_t> bfs_2d(const CSRGraph2D& g, int64_t source, MPI_Comm comm);

#endif
