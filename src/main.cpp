#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfs_1d.h"
#include "bfs_2d.h"
#include "graph_utils.h"

static void usage(const char* prog, std::ostream& out) {
    out << "Usage:\n"
        << "  " << prog << " <graph_file> stats1d [weighted]\n"
        << "  " << prog << " <graph_file> bfs1d <source>\n"
        << "  " << prog << " <graph_file> bfs2d <source> [output_file]\n";
}

static int run_stats1d(const std::string& filename, int rank, int argc, char** argv) {
    bool weighted = (argc >= 4 && std::string(argv[3]) == "weighted");
    CSRGraph full;
    if (rank == 0) full = load_snap_graph_serial(filename, weighted);
    CSRGraph g = distribute_graph_1d(full, MPI_COMM_WORLD);
    print_graph_stats(g, MPI_COMM_WORLD);
    return 0;
}

static int run_bfs1d(const std::string& filename, int rank, int argc, char** argv) {
    if (argc < 4) {
        if (rank == 0) std::cerr << "Error: bfs1d requires <source>\n";
        return 1;
    }
    int64_t source = std::stoll(argv[3]);

    CSRGraph full;
    if (rank == 0) full = load_snap_graph_serial(filename, false);
    CSRGraph g = distribute_graph_1d(full, MPI_COMM_WORLD);

    if (source < 0 || source >= g.n_global) {
        if (rank == 0)
            std::cerr << "Error: source " << source << " out of range [0, "
                      << g.n_global << ")\n";
        return 1;
    }

    std::vector<int64_t> d;
    bfs_1d(g, source, d);
    return 0;
}

static int run_bfs2d(const std::string& filename, int rank, int p, int argc, char** argv) {
    if (argc < 4) {
        if (rank == 0) std::cerr << "Error: bfs2d requires <source>\n";
        return 1;
    }
    int64_t source = std::stoll(argv[3]);
    std::string output_file;
    if (argc >= 5) {
        output_file = argv[4];
    } else {
        // Default: write next to the binary, with a name derived from the graph and source.
        std::string stem = filename;
        size_t slash = stem.find_last_of('/');
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        output_file = "bfs2d_" + stem + "_src" + std::to_string(source) + ".txt";
    }

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

    std::vector<int64_t> parents = bfs_2d(g, source, MPI_COMM_WORLD);

    // Gather all local parent slices onto rank 0. Slices are contiguous and
    // ordered by rank, so a single MPI_Gatherv yields a global parents array.
    int64_t vec_start, vec_end;
    bfs_2d_vec_range(g, &vec_start, &vec_end);
    int local_count = (int)(vec_end - vec_start);

    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(p);
        displs.resize(p);
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int64_t> all_parents;
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < p; i++) {
            displs[i] = total;
            total += counts[i];
        }
        all_parents.resize(total);
    }
    MPI_Gatherv(parents.data(), local_count, MPI_INT64_T,
                all_parents.data(), counts.data(), displs.data(), MPI_INT64_T,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Derive BFS depth from the parent chain. parents[source] is `source`
        // itself (loop sentinel); -1 means unreachable.
        std::vector<int64_t> distance(g.n_global, -1);
        distance[source] = 0;
        for (int64_t v = 0; v < g.n_global; v++) {
            if (distance[v] != -1 || all_parents[v] == -1) continue;
            int64_t cur = v;
            int64_t hops = 0;
            while (distance[cur] == -1) {
                int64_t par = all_parents[cur];
                if (par == -1) { hops = -1; break; }  // unreachable
                cur = par;
                hops++;
            }
            if (hops == -1) continue;
            int64_t base = distance[cur];  // known distance at chain end
            cur = v;
            for (int64_t k = 0; k < hops; k++) {
                distance[cur] = base + (hops - k);
                cur = all_parents[cur];
            }
        }

        std::ofstream fout(output_file);
        if (!fout.is_open()) {
            std::cerr << "Error: cannot open output file: " << output_file << "\n";
            return 1;
        }
        for (int64_t v = 0; v < g.n_global; v++) {
            if (distance[v] == -1) fout << v << " INF\n";
            else                   fout << v << " " << distance[v] << "\n";
        }
        std::cerr << "Wrote BFS distances to " << output_file << "\n";
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
            ret = run_bfs1d(filename, rank, argc, argv);
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
