import numpy as np
import os
import sys

def generate_snap_kronecker(scale, edge_factor=16):
    n = 2**scale
    m = n * edge_factor
    # Kronecker/R-MAT parameters (Graph 500 standard)
    A, B, C = 0.57, 0.19, 0.19
    D = 1.0 - (A + B + C)
    
    ij = np.zeros((2, m), dtype=np.int64)
    ab = A + B
    c_norm = C / (1 - ab)
    a_norm = A / ab

    for i in range(scale):
        ii_bit = np.random.rand(m) > ab
        jj_bit = np.random.rand(m) > (c_norm * ii_bit + a_norm * (~ii_bit))
        ij[0, :] += (2**i) * ii_bit
        ij[1, :] += (2**i) * jj_bit

    # Save in SNAP format (Source Destination)
    out_dir = os.path.join(os.environ["SCRATCH"], "graphs")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"graph_s{scale}.txt")
    np.savetxt(fname, ij.T, fmt='%d', header=f"Nodes: {n} Edges: {m}")
    print(f"Saved {fname}")

if __name__ == "__main__":
    s = int(sys.argv[1])
    generate_snap_kronecker(s)