#include "graph_utils.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Dijkstra ─────────────────────────────────────────────────────────────────

static std::vector<double> dijkstra(const CSRGraph& g, int64_t source,
                                    bool unweighted) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(g.n_global, INF);
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
        g = load_snap_graph_serial(graph_file, !unweighted);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "Vertices: " << g.n_global << "\n";
    std::cerr << "Edges:    " << g.m_global << "\n";
    std::cerr << "Load time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    if (source < 0 || source >= g.n_global) {
        std::cerr << "Error: source " << source << " out of range [0, " << g.n_global << ")\n";
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
    for (int64_t v = 0; v < g.n_global; v++) {
        if (dist[v] == INF)
            out << v << " INF\n";
        else
            out << v << " " << dist[v] << "\n";
    }

    return 0;
}
