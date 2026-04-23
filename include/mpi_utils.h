#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include <vector>
#include <cstdint>

namespace mpi_utils {

// Flat send/recv buffers for MPI_Alltoallv: data is rank-sliced, counts[i] is elements for rank i,
// displs[i] is the prefix-sum offset of rank i's slice.
struct SendBuffer {
    std::vector<int64_t> data;
    std::vector<int>     counts;
    std::vector<int>     displs;
};

struct RecvBuffer {
    std::vector<int64_t> data;
    std::vector<int>     counts;
    std::vector<int>     displs;
};

// Accumulates global vertex IDs per destination rank during BFS/SSSP frontier expansion.
// Call pack() to get a SendBuffer for alltoallv_exchange(), then clear() to reuse next level.
class FrontierPacker {
public:
    explicit FrontierPacker(int p);

    // Append global_v into the bucket for `owner` (must be in [0, p)).
    void add(int64_t global_v, int owner);

    // Flatten per-rank buckets into a SendBuffer; buckets unchanged until clear().
    SendBuffer pack() const;

    // Reset accumulated vertices without deallocating bucket storage.
    void clear();

    int    num_ranks()  const { return p_; }
    size_t total_size() const;

private:
    int p_;
    std::vector<std::vector<int64_t>> buckets_;
};

// Flatten per-rank buckets into a SendBuffer.
SendBuffer build_send_buffer(const std::vector<std::vector<int64_t>>& buckets);

// Exchange counts via MPI_Alltoall and allocate a RecvBuffer; data is uninitialized until do_exchange().
RecvBuffer build_recv_buffer(const SendBuffer& sb, MPI_Comm comm);

// Execute MPI_Alltoallv to fill rb.data from sb.data.
void do_exchange(const SendBuffer& sb, RecvBuffer& rb, MPI_Comm comm);

// One-shot alltoallv: builds buffers, exchanges, and returns the flat received vertex IDs.
std::vector<int64_t> alltoallv_exchange(
    const std::vector<std::vector<int64_t>>& buckets, MPI_Comm comm);

std::vector<int64_t> alltoallv_exchange(
    const SendBuffer& sb, MPI_Comm comm);

} // namespace mpi_utils

#endif // MPI_UTILS_H