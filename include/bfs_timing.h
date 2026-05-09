#ifndef BFS_TIMING_H
#define BFS_TIMING_H

// Per-call BFS wall-clock timing, reported as the maximum across all ranks
// (i.e. the slowest process). compute_time = total_time - comm_time.
struct BFSTiming {
    double total_time;
    double comm_time;
    double compute_time;
};

#endif // BFS_TIMING_H
