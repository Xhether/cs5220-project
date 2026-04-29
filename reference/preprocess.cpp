// Serial preprocessing: load graph, find largest connected component,
// pick 16 random source vertices from the LCC, write info file.
//
// Output format (whitespace-separated key/values):
//   n_lcc <int>
//   m_lcc <int>          # directed edge count, matches CSRGraph::m_global
//   sources <id_0> <id_1> ... <id_15>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct CSRGraph {
    int64_t n;
    int64_t m;
    std::vector<int64_t> offsets;
    std::vector<int64_t> neighbors;
};

static CSRGraph load_graph(const std::string& filename) {
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

    std::sort(all_vertices.begin(), all_vertices.end());
    all_vertices.erase(std::unique(all_vertices.begin(), all_vertices.end()),
                       all_vertices.end());

    std::unordered_map<int64_t,int64_t> remap;
    remap.reserve(all_vertices.size());
    for (int64_t i = 0; i < (int64_t)all_vertices.size(); i++)
        remap[all_vertices[i]] = i;

    int64_t n = (int64_t)all_vertices.size();

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

    CSRGraph g;
    g.n = n;
    g.m = m;
    g.offsets.assign(n + 1, 0);
    for (auto& e : raw_edges) g.offsets[e.first + 1]++;
    for (int64_t i = 1; i <= n; i++) g.offsets[i] += g.offsets[i - 1];

    g.neighbors.resize(m);
    std::vector<int64_t> cursor(g.offsets.begin(), g.offsets.end());
    for (auto& e : raw_edges) g.neighbors[cursor[e.first]++] = e.second;

    return g;
}

// Assigns a component ID to every vertex; returns (comp[], sizes[]).
static std::pair<std::vector<int64_t>, std::vector<int64_t>>
find_components(const CSRGraph& g) {
    std::vector<int64_t> comp(g.n, -1);
    std::vector<int64_t> sizes;
    int64_t next_id = 0;
    std::queue<int64_t> q;
    for (int64_t s = 0; s < g.n; s++) {
        if (comp[s] != -1) continue;
        comp[s] = next_id;
        int64_t count = 0;
        q.push(s);
        while (!q.empty()) {
            int64_t u = q.front(); q.pop();
            count++;
            for (int64_t k = g.offsets[u]; k < g.offsets[u + 1]; k++) {
                int64_t v = g.neighbors[k];
                if (comp[v] == -1) {
                    comp[v] = next_id;
                    q.push(v);
                }
            }
        }
        sizes.push_back(count);
        next_id++;
    }
    return {std::move(comp), std::move(sizes)};
}

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --graph <file> [--output <file>] [--seed <int>] [--num-sources <int>]\n";
}

int main(int argc, char* argv[]) {
    std::string graph_file, output_file = "lcc_info.txt";
    uint64_t seed = 42;
    int num_sources = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--graph") && i + 1 < argc)
            graph_file = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output_file = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = std::stoull(argv[++i]);
        else if (!strcmp(argv[i], "--num-sources") && i + 1 < argc)
            num_sources = std::stoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (graph_file.empty()) { usage(argv[0]); return 1; }

    std::cerr << "Loading graph: " << graph_file << "\n";
    CSRGraph g;
    try {
        g = load_graph(graph_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "Vertices: " << g.n << "\n";
    std::cerr << "Edges:    " << g.m << "  (directed; each undirected counted twice)\n";

    auto [comp, sizes] = find_components(g);
    int64_t num_comps = (int64_t)sizes.size();
    int64_t largest = 0;
    for (int64_t c = 1; c < num_comps; c++)
        if (sizes[c] > sizes[largest]) largest = c;

    int64_t n_lcc = sizes[largest];
    int64_t m_lcc = 0;
    std::vector<int64_t> lcc_vertices;
    lcc_vertices.reserve(n_lcc);
    for (int64_t v = 0; v < g.n; v++) {
        if (comp[v] == largest) {
            lcc_vertices.push_back(v);
            m_lcc += g.offsets[v + 1] - g.offsets[v];
        }
    }
    std::cerr << "Components: " << num_comps << "\n";
    std::cerr << "LCC: n=" << n_lcc << "  m=" << m_lcc
              << "  (" << (100.0 * n_lcc / g.n) << "% of vertices)\n";

    std::mt19937_64 rng(seed);
    std::vector<int64_t> sources;
    if ((int64_t)lcc_vertices.size() <= num_sources) {
        sources = lcc_vertices;
    } else {
        std::shuffle(lcc_vertices.begin(), lcc_vertices.end(), rng);
        sources.assign(lcc_vertices.begin(), lcc_vertices.begin() + num_sources);
    }

    std::ofstream fout(output_file);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot open output file: " << output_file << "\n";
        return 1;
    }
    fout << "n_lcc " << n_lcc << "\n";
    fout << "m_lcc " << m_lcc << "\n";
    fout << "sources";
    for (auto s : sources) fout << " " << s;
    fout << "\n";

    std::cerr << "Wrote LCC info to " << output_file << "\n";
    return 0;
}
