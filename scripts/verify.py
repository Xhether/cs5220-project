import sys
import math

def parse_output(filename):
    """Read a distance output file into a dict {vertex_id: distance}"""
    distances = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vertex = int(parts[0])
            dist = float('inf') if parts[1] == 'INF' else float(parts[1])
            distances[vertex] = dist
    return distances

def compare(serial_file, parallel_file, tolerance=1e-4):
    serial   = parse_output(serial_file)
    parallel = parse_output(parallel_file)

    # Check same number of vertices
    if len(serial) != len(parallel):
        print(f"FAIL: different number of vertices "
              f"(serial={len(serial)}, parallel={len(parallel)})")
        return False

    errors = []
    for v in serial:
        if v not in parallel:
            errors.append(f"  vertex {v} missing from parallel output")
            continue

        s = serial[v]
        p = parallel[v]

        # Both infinite — fine
        if math.isinf(s) and math.isinf(p):
            continue

        # One infinite, one not — wrong
        if math.isinf(s) != math.isinf(p):
            errors.append(f"  vertex {v}: serial={s}, parallel={p}")
            continue

        # Both finite — check within tolerance
        # Use relative tolerance for large distances, absolute for small
        if abs(s - p) > tolerance * max(1.0, abs(s)):
            errors.append(f"  vertex {v}: serial={s:.4f}, parallel={p:.4f}, "
                          f"diff={abs(s-p):.6f}")

    if errors:
        print(f"FAIL: {len(errors)} mismatches found:")
        # Only print first 20 so you're not flooded
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False
    else:
        print(f"CORRECT: all {len(serial)} vertices match")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify.py serial_out.txt parallel_out.txt")
        sys.exit(1)

    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)