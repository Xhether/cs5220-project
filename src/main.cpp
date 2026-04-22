#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include "graph_utils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <graph_file> [weighted]\n";
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    bool weighted = (argc >= 3 && std::string(argv[2]) == "weighted");

    try {
        CSRGraph g = load_snap_graph(filename, weighted, MPI_COMM_WORLD);
        print_graph_stats(g, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << "Error: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
