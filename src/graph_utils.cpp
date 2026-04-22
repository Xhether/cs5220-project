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
#include <mpi.h>

// ─── Ownership / index arithmetic ────────────────────────────────────────────

int owner_of(int64_t global_vertex, int64_t n_global, int p) {
    // Estimate rank via inverse of the block-distribution formula, then clamp
    // and correct for off-by-one at boundaries.
    int r = (int)((global_vertex * (int64_t)p) / n_global);
    if (r >= p) r = p - 1;
    if (r < 0)  r = 0;
    // Nudge right if estimate landed one block too low
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
    // Canonical ordering ensures weight_of(u,v) == weight_of(v,u)
    int64_t lo = (u < v) ? u : v;
    int64_t hi = (u < v) ? v : u;

    // splitmix64-style avalanche
    uint64_t h = (uint64_t)lo * 6364136223846793005ULL
               + (uint64_t)hi * 1442695040888963407ULL;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;

    return (double)(h % 100) + 1.0;
}

// ─── Graph loading ────────────────────────────────────────────────────────────

CSRGraph load_snap_graph(const std::string& filename, bool assign_weights, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    // ── Stage A: every rank reads the file independently ──────────────────────
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

    // ── Stage B: compact vertex remapping + symmetrize + deduplicate ──────────
    // Build 0-based contiguous vertex IDs (SNAP files sometimes have gaps)
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

    // Sort by (src, dst) then deduplicate
    std::sort(raw_edges.begin(), raw_edges.end());
    raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()),
                    raw_edges.end());

    int64_t m_global = (int64_t)raw_edges.size();

    // ── Stage C: each rank builds its own CSR slice ───────────────────────────
    CSRGraph g;
    g.n_global     = n_global;
    g.m_global     = m_global;
    g.vertex_start = ((int64_t)rank * n_global) / p;
    g.vertex_end   = ((int64_t)(rank + 1) * n_global) / p;
    g.n_local      = g.vertex_end - g.vertex_start;
    g.weighted     = assign_weights;

    g.offsets.assign(g.n_local + 1, 0);

    // raw_edges is sorted by src so the local range is a contiguous subarray
    auto lo_it = std::lower_bound(raw_edges.begin(), raw_edges.end(),
                                  std::make_pair(g.vertex_start, (int64_t)0));
    auto hi_it = std::lower_bound(raw_edges.begin(), raw_edges.end(),
                                  std::make_pair(g.vertex_end,   (int64_t)0));

    for (auto it = lo_it; it != hi_it; ++it)
        g.offsets[it->first - g.vertex_start + 1]++;

    for (int64_t i = 1; i <= g.n_local; i++)
        g.offsets[i] += g.offsets[i - 1];

    g.m_local = g.offsets[g.n_local];
    g.neighbors.resize(g.m_local);

    std::vector<int64_t> cursor(g.offsets.begin(), g.offsets.end());
    for (auto it = lo_it; it != hi_it; ++it) {
        int64_t local_u = it->first - g.vertex_start;
        g.neighbors[cursor[local_u]++] = it->second;
    }

    // Neighbors are already sorted within each row (raw_edges sorted by (src,dst))

    // ── Stage D: assign weights locally ──────────────────────────────────────
    if (assign_weights) {
        g.weights.resize(g.m_local);
        for (int64_t local_u = 0; local_u < g.n_local; local_u++) {
            int64_t u_global = local_to_global(local_u, rank, n_global, p);
            for (int64_t j = g.offsets[local_u]; j < g.offsets[local_u + 1]; j++)
                g.weights[j] = weight_of(u_global, g.neighbors[j]);
        }
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

    // Print each rank's local info in order
    for (int r = 0; r < p; r++) {
        MPI_Barrier(comm);
        if (rank == r) {
            printf("  rank %d: vertices [%ld, %ld)  local_edges=%ld\n",
                   rank, (long)g.vertex_start, (long)g.vertex_end, (long)g.m_local);
            fflush(stdout);
        }
    }
}
