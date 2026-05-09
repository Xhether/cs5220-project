#!/usr/bin/env bash
# Strong-scaling experiment for 1D and 2D BFS.
#
# Usage:
#   scripts/strong_scaling.sh <graph_file> <lcc_info_file> <output_csv>
#
# For each (algorithm, num_processes) configuration, picks 5 random sources
# from the LCC info file, runs the BFS once per source, and writes the average
# total/comm/compute/TEPS as one CSV row. Append-safe: if the CSV already
# exists with rows, the header line is not re-emitted.
#
# Must be run inside an allocation (e.g. after `salloc`) so that `srun` works.

set -uo pipefail

GRAPH=${1:?graph file required}
LCC_INFO=${2:?lcc info file required}
CSV=${3:?output csv required}

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN=$PROJECT_DIR/build/parallel_bfs
[ -x "$BIN" ] || { echo "binary not found: $BIN" >&2; exit 1; }

# Parse the 16 sources line from the LCC info file
mapfile -t SOURCES < <(awk '/^sources/ { for (i=2; i<=NF; i++) print $i }' "$LCC_INFO")
[ ${#SOURCES[@]} -ge 2 ] || { echo "need >= 2 sources in $LCC_INFO" >&2; exit 1; }

NUM_RUNS=2

# Write CSV header if the file is new/empty
if [ ! -s "$CSV" ]; then
    echo "algorithm,num_processes,avg_total_time,avg_comm_time,avg_compute_time,avg_teps" > "$CSV"
fi

# Extract value for `key=value` token from a metrics line on stdin.
parse_field() {
    awk -v key="$1" '{
        for (i=1; i<=NF; i++) {
            split($i, kv, "=")
            if (kv[1] == key) { print kv[2]; exit }
        }
    }'
}

run_config() {
    local algo=$1 nproc=$2

    echo "=== $algo p=$nproc [$(date '+%Y-%m-%d %H:%M:%S')] ===" >&2

    # 5 random sources without replacement from the 16 in the LCC info.
    local picks
    mapfile -t picks < <(printf "%s\n" "${SOURCES[@]}" | shuf -n "$NUM_RUNS")

    local sum_total=0 sum_comm=0 sum_compute=0 sum_teps=0
    local n=0

    for src in "${picks[@]}"; do
        local metrics
        metrics=$(srun -n "$nproc" "$BIN" "$GRAPH" "$algo" "$src" \
                       --lcc-info "$LCC_INFO" --no-output 2>/dev/null \
                       | grep "^\[$algo\]" || true)
        if [ -z "$metrics" ]; then
            echo "  $algo @ p=$nproc src=$src: no metrics line (run failed?)" >&2
            continue
        fi

        local total comm cmpu teps
        total=$(echo "$metrics" | parse_field total_time)
        comm=$(echo  "$metrics" | parse_field comm_time)
        cmpu=$(echo  "$metrics" | parse_field compute_time)
        teps=$(echo  "$metrics" | parse_field teps)

        echo "  $algo @ p=$nproc src=$src: total=$total comm=$comm compute=$cmpu teps=$teps" >&2

        sum_total=$(awk   -v a="$sum_total"   -v b="$total" 'BEGIN{print a+b}')
        sum_comm=$(awk    -v a="$sum_comm"    -v b="$comm"  'BEGIN{print a+b}')
        sum_compute=$(awk -v a="$sum_compute" -v b="$cmpu"  'BEGIN{print a+b}')
        sum_teps=$(awk    -v a="$sum_teps"    -v b="$teps"  'BEGIN{print a+b}')
        n=$((n+1))
    done

    if [ "$n" -eq 0 ]; then
        echo "$algo @ p=$nproc: no successful runs, skipping CSV row" >&2
        return
    fi

    local avg_total avg_comm avg_compute avg_teps
    avg_total=$(awk   -v a="$sum_total"   -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_comm=$(awk    -v a="$sum_comm"    -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_compute=$(awk -v a="$sum_compute" -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_teps=$(awk    -v a="$sum_teps"    -v n="$n" 'BEGIN{printf "%.6e", a/n}')

    local row="$algo,$nproc,$avg_total,$avg_comm,$avg_compute,$avg_teps"
    echo "$row"
    echo "$row" >> "$CSV"
}

# 1D BFS strong scaling
for P in 1 2 4 8 16 32 64 128 256; do
    run_config bfs1d "$P"
done

# 2D BFS strong scaling (square ranks only)
for P in 1 4 16 64 256; do
    run_config bfs2d "$P"
done
