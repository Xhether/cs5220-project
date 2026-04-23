cmake_minimum_required(VERSION 3.16)
project(ParallelGraphAlgorithms)

# Use C++17 for modern features and CSR implementation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI for distributed memory processing [cite: 25, 96]
find_package(MPI REQUIRED)

# List all source files
# This includes your 1D and 2D partitioning implementations 
set(SOURCES
    src/main.cpp
    src/bfs_1d.cpp
    src/bfs_2d.cpp
    src/sssp_1d.cpp     
    src/sssp_2d.cpp
    src/graph_utils.cpp
    src/mpi_utils.cpp
)

# Define the executable
add_executable(parallel_bfs ${SOURCES})

# Tell the compiler where your header files are
target_include_directories(parallel_bfs PRIVATE include)

# Link the MPI library [cite: 96, 99]
target_link_libraries(parallel_bfs PRIVATE MPI::MPI_CXX)

# Optimization and Warning Flags
# -O3: Max optimization for TEPS and scaling analysis 
# -Wall -Wextra: Enable warnings to catch bugs early [cite: 195, 196]
target_compile_options(parallel_bfs PRIVATE 
    -O3 
    -Wall 
    -Wextra 
)

# serial reference — no MPI, just plain C++
add_executable(serial_reference
    reference/dijkstra.cpp
    src/graph_utils.cpp
)

target_include_directories(serial_reference PRIVATE include)
target_link_libraries(serial_reference PRIVATE MPI::MPI_CXX)
target_compile_options(serial_reference PRIVATE
    -O3
    -Wall
    -Wextra
)