#include "graph_utils.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <stdexcept>
#include <climits>
#include <cstdint>
#include <cstdio>

// ─── Ownership / index arithmetic ────────────────────────────────────────────
// what rank owns vertex global_vertex given n_global vertices and p ranks 
int owner_of(int64_t global_vertex, int64_t n_global, int p) {
    int r = (int)((global_vertex * (int64_t)p) / n_global);
    if (r >= p) r = p - 1;
    if (r < 0)  r = 0;
    while (r + 1 < p && ((int64_t)(r + 1) * n_global) / p <= global_vertex) r++;
    return r;
}

int64_t global_to_local(int64_t global_vertex, int64_t n_global, int p) {
    int r = owner_of(global_vertex, n_global, p);
    int64_t start = ((int64_t)r * n_global) / p;
    return global_vertex - start;
}

int64_t local_to_global(int64_t local_index, int rank, int64_t n_global, int p) {
    int64_t start = ((int64_t)rank * n_global) / p;
    return start + local_index;
}

// ─── Deterministic edge weight ────────────────────────────────────────────────

double weight_of(int64_t u, int64_t v) {
    int64_t lo = (u < v) ? u : v;
    int64_t hi = (u < v) ? v : u;

    uint64_t h = (uint64_t)lo * 6364136223846793005ULL
               + (uint64_t)hi * 1442695040888963407ULL;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;

    return (double)(h % 100) + 1.0;
}

// ─── Stage 1: serial graph loading (rank 0 only, no MPI) ─────────────────────

CSRGraph load_snap_graph_serial(const std::string& filename, bool assign_weights) {
    std::ifstream fin(filename);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open graph file: " + filename);

    std::vector<std::pair<int64_t,int64_t>> raw_edges;
    std::vector<int64_t> all_vertices;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int64_t u, v;
        if (!(ss >> u >> v)) continue;
        if (u == v) continue; // skip self-loops
        raw_edges.push_back({u, v});
        all_vertices.push_back(u);
        all_vertices.push_back(v);
    }
    fin.close();

    // Compact vertex IDs to 0..n-1 (SNAP files often have gaps)
    std::sort(all_vertices.begin(), all_vertices.end());
    all_vertices.erase(std::unique(all_vertices.begin(), all_vertices.end()),
                       all_vertices.end());

    std::unordered_map<int64_t,int64_t> remap;
    remap.reserve(all_vertices.size());
    for (int64_t i = 0; i < (int64_t)all_vertices.size(); i++)
        remap[all_vertices[i]] = i;

    int64_t n_global = (int64_t)all_vertices.size();
    all_vertices.clear();
    all_vertices.shrink_to_fit();

    // Remap and symmetrize
    size_t orig = raw_edges.size();
    raw_edges.reserve(orig * 2);
    for (size_t i = 0; i < orig; i++) {
        int64_t u = remap[raw_edges[i].first];
        int64_t v = remap[raw_edges[i].second];
        raw_edges[i] = {u, v};
        raw_edges.push_back({v, u});
    }

    std::sort(raw_edges.begin(), raw_edges.end());
    raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()),
                    raw_edges.end());

    int64_t m_global = (int64_t)raw_edges.size();

    // Build a single-"rank" CSR covering all vertices
    CSRGraph g;
    g.n_global     = n_global;
    g.m_global     = m_global;
    g.vertex_start = 0;
    g.vertex_end   = n_global;
    g.n_local      = n_global;
    g.m_local      = m_global;
    g.weighted     = assign_weights;

    g.offsets.assign(n_global + 1, 0);
    for (const auto& e : raw_edges) g.offsets[e.first + 1]++;
    for (int64_t i = 1; i <= n_global; i++) g.offsets[i] += g.offsets[i - 1];

    g.neighbors.resize(m_global);
    // raw_edges is already sorted by (src, dst), so we can just copy dsts
    for (int64_t i = 0; i < m_global; i++)
        g.neighbors[i] = raw_edges[i].second;

    if (assign_weights) {
        g.weights.resize(m_global);
        for (int64_t u = 0; u < n_global; u++) {
            for (int64_t j = g.offsets[u]; j < g.offsets[u + 1]; j++)
                g.weights[j] = weight_of(u, g.neighbors[j]);
        }
    }

    return g;
}

// ─── Stage 2a: 1D row-block distribution ─────────────────────────────────────

CSRGraph distribute_graph_1d(const CSRGraph& full, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // Broadcast graph-wide metadata from rank 0
    int64_t meta[3];
    if (rank == 0) {
        meta[0] = full.n_global;
        meta[1] = full.m_global;
        meta[2] = full.weighted ? 1 : 0;
    }
    MPI_Bcast(meta, 3, MPI_INT64_T, 0, comm);

    CSRGraph g;
    g.n_global     = meta[0];
    g.m_global     = meta[1];
    g.weighted     = (meta[2] != 0);
    g.vertex_start = ((int64_t)rank * g.n_global) / p;
    g.vertex_end   = ((int64_t)(rank + 1) * g.n_global) / p;
    g.n_local      = g.vertex_end - g.vertex_start;

    // Scatter the CSR offsets (n_local + 1 entries per rank, with overlap at boundaries)
    std::vector<int> off_counts, off_displs;
    std::vector<int> edge_counts, edge_displs;
    if (rank == 0) {
        off_counts.resize(p);
        off_displs.resize(p);
        edge_counts.resize(p);
        edge_displs.resize(p);
        for (int r = 0; r < p; r++) {
            int64_t vs = ((int64_t)r * g.n_global) / p;
            int64_t ve = ((int64_t)(r + 1) * g.n_global) / p;
            off_displs[r]  = (int)vs;
            off_counts[r]  = (int)(ve - vs + 1);   // +1 to include the sentinel
            edge_displs[r] = (int)full.offsets[vs];
            edge_counts[r] = (int)(full.offsets[ve] - full.offsets[vs]);
        }
    }

    // Receive our offset slice (n_local + 1 entries)
    std::vector<int64_t> raw_offsets(g.n_local + 1);
    MPI_Scatterv(rank == 0 ? full.offsets.data() : nullptr,
                 off_counts.data(), off_displs.data(), MPI_INT64_T,
                 raw_offsets.data(), (int)(g.n_local + 1), MPI_INT64_T,
                 0, comm);

    // Rebase offsets so local_offsets[0] == 0
    int64_t base = raw_offsets[0];
    g.offsets.resize(g.n_local + 1);
    for (int64_t i = 0; i <= g.n_local; i++)
        g.offsets[i] = raw_offsets[i] - base;
    g.m_local = g.offsets[g.n_local];

    // Scatter neighbors
    g.neighbors.resize(g.m_local);
    MPI_Scatterv(rank == 0 ? full.neighbors.data() : nullptr,
                 edge_counts.data(), edge_displs.data(), MPI_INT64_T,
                 g.neighbors.data(), (int)g.m_local, MPI_INT64_T,
                 0, comm);

    // Scatter weights if present
    if (g.weighted) {
        g.weights.resize(g.m_local);
        MPI_Scatterv(rank == 0 ? full.weights.data() : nullptr,
                     edge_counts.data(), edge_displs.data(), MPI_DOUBLE,
                     g.weights.data(), (int)g.m_local, MPI_DOUBLE,
                     0, comm);
    }

    return g;
}

// ─── Stage 2b: 2D checkerboard distribution ──────────────────────────────────

CSRGraph2D distribute_graph_2d(const CSRGraph& full, int grid_rows, int grid_cols, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    if (grid_rows * grid_cols != p)
        throw std::runtime_error("distribute_graph_2d: grid_rows * grid_cols must equal comm size");

    // Broadcast metadata
    int64_t meta[3];
    if (rank == 0) {
        meta[0] = full.n_global;
        meta[1] = full.m_global;
        meta[2] = full.weighted ? 1 : 0;
    }
    MPI_Bcast(meta, 3, MPI_INT64_T, 0, comm);

    CSRGraph2D g;
    g.n_global  = meta[0];
    g.m_global  = meta[1];
    g.weighted  = (meta[2] != 0);
    g.grid_rows = grid_rows;
    g.grid_cols = grid_cols;
    g.pr        = rank / grid_cols;
    g.pc        = rank % grid_cols;
    g.row_start = ((int64_t)g.pr * g.n_global) / grid_rows;
    g.row_end   = ((int64_t)(g.pr + 1) * g.n_global) / grid_rows;
    g.col_start = ((int64_t)g.pc * g.n_global) / grid_cols;
    g.col_end   = ((int64_t)(g.pc + 1) * g.n_global) / grid_cols;
    g.n_local_rows = g.row_end - g.row_start;

    // Rank 0 builds every tile's (offsets, neighbors, weights) then sends to the
    // owning rank. We keep tile 0 locally and MPI_Send the others.
    //
    // Why build-then-send rather than Scatterv: tiles for different ranks are
    // not contiguous in the full CSR (a row's neighbors get split across column
    // bands), so there's no single contiguous buffer to scatter from.

    auto build_tile = [&](int tr, int tc,
                          std::vector<int64_t>& t_offsets,
                          std::vector<int64_t>& t_neighbors,
                          std::vector<double>&  t_weights) {
        int64_t rs = ((int64_t)tr * g.n_global) / grid_rows;
        int64_t re = ((int64_t)(tr + 1) * g.n_global) / grid_rows;
        int64_t cs = ((int64_t)tc * g.n_global) / grid_cols;
        int64_t ce = ((int64_t)(tc + 1) * g.n_global) / grid_cols;
        int64_t nr = re - rs;

        t_offsets.assign(nr + 1, 0);
        // First pass: count per-row edges in the column band.
        // Neighbors within a row are sorted by global v, so we can binary-search.
        for (int64_t u = 0; u < nr; u++) {
            int64_t u_g = rs + u;
            auto beg = full.neighbors.begin() + full.offsets[u_g];
            auto end = full.neighbors.begin() + full.offsets[u_g + 1];
            auto lo = std::lower_bound(beg, end, cs);
            auto hi = std::lower_bound(beg, end, ce);
            t_offsets[u + 1] = (int64_t)(hi - lo);
        }
        for (int64_t u = 1; u <= nr; u++) t_offsets[u] += t_offsets[u - 1];
        int64_t m_tile = t_offsets[nr];

        t_neighbors.resize(m_tile);
        if (g.weighted) t_weights.resize(m_tile);

        for (int64_t u = 0; u < nr; u++) {
            int64_t u_g = rs + u;
            auto beg = full.neighbors.begin() + full.offsets[u_g];
            auto end = full.neighbors.begin() + full.offsets[u_g + 1];
            auto lo = std::lower_bound(beg, end, cs);
            auto hi = std::lower_bound(beg, end, ce);
            int64_t out = t_offsets[u];
            for (auto it = lo; it != hi; ++it, ++out) {
                t_neighbors[out] = *it;
                if (g.weighted) {
                    int64_t j = (it - full.neighbors.begin());
                    t_weights[out] = full.weights[j];
                }
            }
        }
    };

    if (rank == 0) {
        for (int r = 0; r < p; r++) {
            int tr = r / grid_cols;
            int tc = r % grid_cols;
            std::vector<int64_t> t_offsets, t_neighbors;
            std::vector<double>  t_weights;
            build_tile(tr, tc, t_offsets, t_neighbors, t_weights);

            if (r == 0) {
                g.offsets   = std::move(t_offsets);
                g.neighbors = std::move(t_neighbors);
                g.weights   = std::move(t_weights);
                g.m_local   = (int64_t)g.neighbors.size();
            } else {
                int64_t sizes[2] = { (int64_t)t_offsets.size(),
                                     (int64_t)t_neighbors.size() };
                MPI_Send(sizes, 2, MPI_INT64_T, r, 100, comm);
                MPI_Send(t_offsets.data(), (int)t_offsets.size(),
                         MPI_INT64_T, r, 101, comm);
                MPI_Send(t_neighbors.data(), (int)t_neighbors.size(),
                         MPI_INT64_T, r, 102, comm);
                if (g.weighted) {
                    MPI_Send(t_weights.data(), (int)t_weights.size(),
                             MPI_DOUBLE, r, 103, comm);
                }
            }
        }
    } else {
        int64_t sizes[2];
        MPI_Recv(sizes, 2, MPI_INT64_T, 0, 100, comm, MPI_STATUS_IGNORE);
        g.offsets.resize(sizes[0]);
        g.neighbors.resize(sizes[1]);
        MPI_Recv(g.offsets.data(), (int)sizes[0], MPI_INT64_T, 0, 101,
                 comm, MPI_STATUS_IGNORE);
        MPI_Recv(g.neighbors.data(), (int)sizes[1], MPI_INT64_T, 0, 102,
                 comm, MPI_STATUS_IGNORE);
        if (g.weighted) {
            g.weights.resize(sizes[1]);
            MPI_Recv(g.weights.data(), (int)sizes[1], MPI_DOUBLE, 0, 103,
                     comm, MPI_STATUS_IGNORE);
        }
        g.m_local = (int64_t)g.neighbors.size();
    }

    return g;
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

void print_graph_stats(const CSRGraph& g, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    int64_t local_min = INT64_MAX, local_max = 0, local_sum = 0;
    for (int64_t v = 0; v < g.n_local; v++) {
        int64_t deg = local_degree(g, v);
        if (deg < local_min) local_min = deg;
        if (deg > local_max) local_max = deg;
        local_sum += deg;
    }
    if (g.n_local == 0) local_min = 0;

    int64_t global_min, global_max, global_sum;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, 0, comm);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, 0, comm);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, 0, comm);

    if (rank == 0) {
        double avg = (g.n_global > 0) ? (double)global_sum / g.n_global : 0.0;
        printf("Graph: n=%ld vertices, m=%ld edges, weighted=%s\n",
               (long)g.n_global, (long)g.m_global, g.weighted ? "yes" : "no");
        printf("Degree: min=%ld  max=%ld  avg=%.2f\n",
               (long)global_min, (long)global_max, avg);
        printf("Distribution across %d ranks:\n", p);
        fflush(stdout);
    }

    for (int r = 0; r < p; r++) {
        MPI_Barrier(comm);
        if (rank == r) {
            printf("  rank %d: vertices [%ld, %ld)  local_edges=%ld\n",
                   rank, (long)g.vertex_start, (long)g.vertex_end, (long)g.m_local);
            fflush(stdout);
        }
    }
}
