#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfs_1d.h"
#include "bfs_2d.h"
#include "bfs_timing.h"
#include "graph_utils.h"

static void usage(const char* prog, std::ostream& out) {
    out << "Usage:\n"
        << "  " << prog << " <graph_file> stats1d [weighted]\n"
        << "  " << prog << " <graph_file> bfs1d <source> [--lcc-info <file>] [--output <path>] [--no-output]\n"
        << "  " << prog << " <graph_file> bfs2d <source> [--lcc-info <file>] [--output <path>] [--no-output]\n";
}

struct BFSOpts {
    std::string lcc_info;
    std::string output_file;        // "" if unset
    bool        output_set = false; // explicit --output
    bool        no_output  = false;
};

// Parse flags after the source positional. argv[start..argc) is searched.
static bool parse_bfs_flags(int argc, char** argv, int start, BFSOpts& opts, int rank) {
    for (int i = start; i < argc; i++) {
        if (!std::strcmp(argv[i], "--lcc-info") && i + 1 < argc) {
            opts.lcc_info = argv[++i];
        } else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) {
            opts.output_file = argv[++i];
            opts.output_set  = true;
        } else if (!std::strcmp(argv[i], "--no-output")) {
            opts.no_output = true;
        } else {
            if (rank == 0) std::cerr << "Error: unknown argument '" << argv[i] << "'\n";
            return false;
        }
    }
    return true;
}

// Reads the m_lcc value from the preprocessing file. Returns -1 on failure.
static int64_t read_m_lcc(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) return -1;
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "m_lcc") {
            int64_t v;
            if (ss >> v) return v;
        }
    }
    return -1;
}

// Default output path "<algo>_<graph_stem>_src<source>.txt" in cwd.
static std::string default_output_path(const std::string& algo,
                                       const std::string& graph_file,
                                       int64_t source) {
    std::string stem = graph_file;
    size_t slash = stem.find_last_of('/');
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    return algo + "_" + stem + "_src" + std::to_string(source) + ".txt";
}

static void print_metrics(const std::string& algo, int64_t source, int p,
                          int64_t n_lcc, int64_t m_lcc, const BFSTiming& t) {
    double teps = (t.total_time > 0.0) ? (double)m_lcc / t.total_time : 0.0;
    std::cout << "[" << algo << "] source=" << source
              << " ranks=" << p
              << " n_lcc=" << n_lcc
              << " m_lcc=" << m_lcc
              << " total_time=" << t.total_time
              << " comm_time=" << t.comm_time
              << " compute_time=" << t.compute_time
              << " teps=" << teps
              << "\n";
    std::cout.flush();
}

static int run_stats1d(const std::string& filename, int rank, int argc, char** argv) {
    bool weighted = (argc >= 4 && std::string(argv[3]) == "weighted");
    CSRGraph full;
    if (rank == 0) full = load_snap_graph_serial(filename, weighted);
    CSRGraph g = distribute_graph_1d(full, MPI_COMM_WORLD);
    print_graph_stats(g, MPI_COMM_WORLD);
    return 0;
}

static int run_bfs1d(const std::string& filename, int rank, int p, int argc, char** argv) {
    if (argc < 4) {
        if (rank == 0) std::cerr << "Error: bfs1d requires <source>\n";
        return 1;
    }
    int64_t source = std::stoll(argv[3]);

    BFSOpts opts;
    if (!parse_bfs_flags(argc, argv, 4, opts, rank)) return 1;

    CSRGraph full;
    if (rank == 0) full = load_snap_graph_serial(filename, false);
    CSRGraph g = distribute_graph_1d(full, MPI_COMM_WORLD);

    if (source < 0 || source >= g.n_global) {
        if (rank == 0)
            std::cerr << "Error: source " << source << " out of range [0, "
                      << g.n_global << ")\n";
        return 1;
    }

    int64_t m_lcc = g.m_global, n_lcc = g.n_global;
    if (!opts.lcc_info.empty()) {
        int64_t v = read_m_lcc(opts.lcc_info);
        if (v < 0) {
            if (rank == 0) std::cerr << "Error: failed to read m_lcc from " << opts.lcc_info << "\n";
            return 1;
        }
        m_lcc = v;
    }

    std::vector<int64_t> d;
    BFSTiming timing{};
    bfs_1d(g, source, d, &timing);

    if (rank == 0) print_metrics("bfs1d", source, p, n_lcc, m_lcc, timing);

    if (opts.no_output) return 0;

    // Gather local distance slices on rank 0 and write the output file.
    int local_count = (int)g.n_local;
    std::vector<int> counts, displs;
    if (rank == 0) { counts.resize(p); displs.resize(p); }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int64_t> all_d;
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < p; i++) { displs[i] = total; total += counts[i]; }
        all_d.resize(total);
    }
    MPI_Gatherv(d.data(), local_count, MPI_INT64_T,
                all_d.data(), counts.data(), displs.data(), MPI_INT64_T,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::string output = opts.output_set ? opts.output_file
                                             : default_output_path("bfs1d", filename, source);
        std::ofstream fout(output);
        if (!fout.is_open()) {
            std::cerr << "Error: cannot open output file: " << output << "\n";
            return 1;
        }
        const int64_t INF = std::numeric_limits<int64_t>::max();
        for (int64_t v = 0; v < g.n_global; v++) {
            if (all_d[v] == INF) fout << v << " INF\n";
            else                 fout << v << " " << all_d[v] << "\n";
        }
        std::cerr << "Wrote BFS distances to " << output << "\n";
    }
    return 0;
}

static int run_bfs2d(const std::string& filename, int rank, int p, int argc, char** argv) {
    if (argc < 4) {
        if (rank == 0) std::cerr << "Error: bfs2d requires <source>\n";
        return 1;
    }
    int64_t source = std::stoll(argv[3]);

    BFSOpts opts;
    if (!parse_bfs_flags(argc, argv, 4, opts, rank)) return 1;

    int R = (int)std::lround(std::sqrt((double)p));
    if (R * R != p) {
        if (rank == 0)
            std::cerr << "Error: bfs2d requires a square number of ranks (got " << p << ")\n";
        return 1;
    }

    CSRGraph full;
    if (rank == 0) full = load_snap_graph_serial(filename, false);
    CSRGraph2D g = distribute_graph_2d(full, R, R, MPI_COMM_WORLD);

    if (source < 0 || source >= g.n_global) {
        if (rank == 0)
            std::cerr << "Error: source " << source << " out of range [0, "
                      << g.n_global << ")\n";
        return 1;
    }

    int64_t m_lcc = g.m_global, n_lcc = g.n_global;
    if (!opts.lcc_info.empty()) {
        int64_t v = read_m_lcc(opts.lcc_info);
        if (v < 0) {
            if (rank == 0) std::cerr << "Error: failed to read m_lcc from " << opts.lcc_info << "\n";
            return 1;
        }
        m_lcc = v;
    }

    BFSTiming timing{};
    std::vector<int64_t> parents = bfs_2d(g, source, MPI_COMM_WORLD, &timing);

    if (rank == 0) print_metrics("bfs2d", source, p, n_lcc, m_lcc, timing);

    if (opts.no_output) return 0;

    // Gather local parent slices on rank 0 (ordered by rank tiles [0, n_global)).
    int64_t vec_start, vec_end;
    bfs_2d_vec_range(g, &vec_start, &vec_end);
    int local_count = (int)(vec_end - vec_start);

    std::vector<int> counts, displs;
    if (rank == 0) { counts.resize(p); displs.resize(p); }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int64_t> all_parents;
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < p; i++) { displs[i] = total; total += counts[i]; }
        all_parents.resize(total);
    }
    MPI_Gatherv(parents.data(), local_count, MPI_INT64_T,
                all_parents.data(), counts.data(), displs.data(), MPI_INT64_T,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Derive BFS depth from the parent chain. parents[source] == source (loop sentinel).
        std::vector<int64_t> distance(g.n_global, -1);
        distance[source] = 0;
        for (int64_t v = 0; v < g.n_global; v++) {
            if (distance[v] != -1 || all_parents[v] == -1) continue;
            int64_t cur = v;
            int64_t hops = 0;
            while (distance[cur] == -1) {
                int64_t par = all_parents[cur];
                if (par == -1) { hops = -1; break; }
                cur = par;
                hops++;
            }
            if (hops == -1) continue;
            int64_t base = distance[cur];
            cur = v;
            for (int64_t k = 0; k < hops; k++) {
                distance[cur] = base + (hops - k);
                cur = all_parents[cur];
            }
        }

        std::string output = opts.output_set ? opts.output_file
                                             : default_output_path("bfs2d", filename, source);
        std::ofstream fout(output);
        if (!fout.is_open()) {
            std::cerr << "Error: cannot open output file: " << output << "\n";
            return 1;
        }
        for (int64_t v = 0; v < g.n_global; v++) {
            if (distance[v] == -1) fout << v << " INF\n";
            else                   fout << v << " " << distance[v] << "\n";
        }
        std::cerr << "Wrote BFS distances to " << output << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 3) {
        if (rank == 0) usage(argv[0], std::cerr);
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    std::string mode     = argv[2];

    int ret = 0;
    try {
        if (mode == "stats1d")
            ret = run_stats1d(filename, rank, argc, argv);
        else if (mode == "bfs1d")
            ret = run_bfs1d(filename, rank, p, argc, argv);
        else if (mode == "bfs2d")
            ret = run_bfs2d(filename, rank, p, argc, argv);
        else {
            if (rank == 0) {
                std::cerr << "Error: unknown mode '" << mode << "'\n";
                usage(argv[0], std::cerr);
            }
            ret = 1;
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Error: " << e.what() << "\n";
        ret = 1;
    }

    MPI_Finalize();
    return ret;
}
