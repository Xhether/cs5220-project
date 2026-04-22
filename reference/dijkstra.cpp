#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─── Weight function (mirrors graph_utils.cpp exactly) ───────────────────────

static double weight_of(int64_t u, int64_t v) {
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

// ─── CSR graph (single-rank, no MPI) ─────────────────────────────────────────

struct CSRGraph {
    int64_t n;                      // vertex count
    int64_t m;                      // edge count (both directions)
    std::vector<int64_t> offsets;   // size n+1
    std::vector<int64_t> neighbors; // size m
    std::vector<double>  weights;   // size m; empty when unweighted
};

// Reads SNAP edge list, symmetrizes, deduplicates, remaps to 0-based IDs,
// and builds a CSR. Matches the logic in load_snap_graph from graph_utils.cpp.
static CSRGraph load_graph(const std::string& filename, bool assign_weights) {
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
        if (u == v) continue;
        raw_edges.push_back({u, v});
        all_vertices.push_back(u);
        all_vertices.push_back(v);
    }

    // Compact vertex remapping (SNAP files can have gaps)
    std::sort(all_vertices.begin(), all_vertices.end());
    all_vertices.erase(std::unique(all_vertices.begin(), all_vertices.end()),
                       all_vertices.end());

    std::unordered_map<int64_t,int64_t> remap;
    remap.reserve(all_vertices.size());
    for (int64_t i = 0; i < (int64_t)all_vertices.size(); i++)
        remap[all_vertices[i]] = i;

    int64_t n = (int64_t)all_vertices.size();

    // Remap + symmetrize
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

    int64_t m = (int64_t)raw_edges.size();

    // Build CSR
    CSRGraph g;
    g.n = n;
    g.m = m;
    g.offsets.assign(n + 1, 0);

    for (auto& [u, v] : raw_edges)
        g.offsets[u + 1]++;
    for (int64_t i = 1; i <= n; i++)
        g.offsets[i] += g.offsets[i - 1];

    g.neighbors.resize(m);
    std::vector<int64_t> cursor(g.offsets.begin(), g.offsets.end());
    for (auto& [u, v] : raw_edges)
        g.neighbors[cursor[u]++] = v;

    if (assign_weights) {
        g.weights.resize(m);
        for (int64_t u = 0; u < n; u++)
            for (int64_t j = g.offsets[u]; j < g.offsets[u + 1]; j++)
                g.weights[j] = weight_of(u, g.neighbors[j]);
    }

    return g;
}

// ─── Dijkstra ─────────────────────────────────────────────────────────────────

static std::vector<double> dijkstra(const CSRGraph& g, int64_t source,
                                    bool unweighted) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(g.n, INF);
    dist[source] = 0.0;

    using Entry = std::pair<double, int64_t>; // (dist, vertex)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    pq.push({0.0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue; // stale entry

        for (int64_t j = g.offsets[u]; j < g.offsets[u + 1]; j++) {
            int64_t v = g.neighbors[j];
            double w = unweighted ? 1.0 : g.weights[j];
            double nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    return dist;
}

// ─── main ─────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --graph <file> --source <int> [--unweighted] [--output <file>]\n";
}

int main(int argc, char* argv[]) {
    std::string graph_file, output_file;
    int64_t source = 0;
    bool unweighted = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--graph") && i + 1 < argc)
            graph_file = argv[++i];
        else if (!strcmp(argv[i], "--source") && i + 1 < argc)
            source = std::stoll(argv[++i]);
        else if (!strcmp(argv[i], "--unweighted"))
            unweighted = true;
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output_file = argv[++i];
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (graph_file.empty()) {
        usage(argv[0]);
        return 1;
    }

    // Load
    std::cerr << "Loading graph: " << graph_file << "\n";
    auto t0 = std::chrono::steady_clock::now();

    CSRGraph g;
    try {
        g = load_graph(graph_file, !unweighted);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "Vertices: " << g.n << "\n";
    std::cerr << "Edges:    " << g.m << "\n";
    std::cerr << "Load time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    if (source < 0 || source >= g.n) {
        std::cerr << "Error: source " << source << " out of range [0, " << g.n << ")\n";
        return 1;
    }

    // Run Dijkstra
    std::cerr << "Running Dijkstra from source " << source << " ("
              << (unweighted ? "unweighted" : "weighted") << ")\n";
    auto t2 = std::chrono::steady_clock::now();

    std::vector<double> dist = dijkstra(g, source, unweighted);

    auto t3 = std::chrono::steady_clock::now();
    std::cerr << "Dijkstra runtime: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";

    // Output results
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream fout;
    if (!output_file.empty()) {
        fout.open(output_file);
        if (!fout.is_open()) {
            std::cerr << "Error: cannot open output file: " << output_file << "\n";
            return 1;
        }
        buf = fout.rdbuf();
    }
    std::ostream out(buf);

    const double INF = std::numeric_limits<double>::infinity();
    for (int64_t v = 0; v < g.n; v++) {
        if (dist[v] == INF)
            out << v << " INF\n";
        else
            out << v << " " << dist[v] << "\n";
    }

    return 0;
}
