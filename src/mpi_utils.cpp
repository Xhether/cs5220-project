#include "mpi_utils.h"

#include <algorithm>

namespace mpi_utils {

FrontierPacker::FrontierPacker(int p) : p_(p), buckets_(p) {}

void FrontierPacker::add(int64_t global_v, int owner) {
    buckets_[owner].push_back(global_v);
}

SendBuffer FrontierPacker::pack() const {
    return build_send_buffer(buckets_);
}

void FrontierPacker::clear() {
    for (auto& b : buckets_) b.clear();
}

size_t FrontierPacker::total_size() const {
    size_t total = 0;
    for (const auto& b : buckets_) total += b.size();
    return total;
}

SendBuffer build_send_buffer(const std::vector<std::vector<int64_t>>& buckets) {
    const int p = static_cast<int>(buckets.size());
    SendBuffer sb;
    sb.counts.resize(p);
    sb.displs.resize(p);

    size_t total = 0;
    for (int i = 0; i < p; ++i) {
        sb.counts[i] = static_cast<int>(buckets[i].size());
        sb.displs[i] = static_cast<int>(total);
        total += buckets[i].size();
    }

    sb.data.resize(total);
    for (int i = 0; i < p; ++i) {
        std::copy(buckets[i].begin(), buckets[i].end(),
                  sb.data.begin() + sb.displs[i]);
    }
    return sb;
}

RecvBuffer build_recv_buffer(const SendBuffer& sb, MPI_Comm comm) {
    int p;
    MPI_Comm_size(comm, &p);

    RecvBuffer rb;
    rb.counts.resize(p);
    rb.displs.resize(p);

    MPI_Alltoall(sb.counts.data(), 1, MPI_INT,
                 rb.counts.data(), 1, MPI_INT, comm);

    size_t total = 0;
    for (int i = 0; i < p; ++i) {
        rb.displs[i] = static_cast<int>(total);
        total += rb.counts[i];
    }
    rb.data.resize(total);
    return rb;
}

void do_exchange(const SendBuffer& sb, RecvBuffer& rb, MPI_Comm comm) {
    MPI_Alltoallv(sb.data.data(), sb.counts.data(), sb.displs.data(), MPI_INT64_T,
                  rb.data.data(), rb.counts.data(), rb.displs.data(), MPI_INT64_T,
                  comm);
}

std::vector<int64_t> alltoallv_exchange(const SendBuffer& sb, MPI_Comm comm) {
    RecvBuffer rb = build_recv_buffer(sb, comm);
    do_exchange(sb, rb, comm);
    return std::move(rb.data);
}

std::vector<int64_t> alltoallv_exchange(
    const std::vector<std::vector<int64_t>>& buckets, MPI_Comm comm) {
    SendBuffer sb = build_send_buffer(buckets);
    return alltoallv_exchange(sb, comm);
}

} // namespace mpi_utils
