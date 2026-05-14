#!/usr/bin/env bash
# Strong-scaling experiment for 1D and 2D delta-stepping SSSP.
#
# Usage:
#   scripts/sssp_benchmark.sh <graph_file> <lcc_info_file> <output_csv> [delta]
#
# Examples (run inside an `salloc` allocation on Perlmutter):
#   scripts/sssp_benchmark.sh \
#       $SCRATCH/graphs/com-lj.ungraph.txt   com-lj_lcc.txt        sssp_lj.csv
#   scripts/sssp_benchmark.sh \
#       $SCRATCH/graphs/roadNet-CA.txt       roadNet-CA_LCC.txt    sssp_road.csv
#   scripts/sssp_benchmark.sh \
#       $SCRATCH/graphs/ca-GrQc.txt          ca-GrQc_lcc.txt       sssp_grqc.csv
#
# For each (algorithm, num_processes) configuration, picks NUM_RUNS random
# sources from the LCC info file, runs SSSP once per source, and writes the
# average total_time / TEPS as one CSV row. Append-safe: if the CSV already
# exists with rows, the header line is not re-emitted.
#
# This intentionally mirrors strong_scaling.sh so SSSP and BFS scaling runs
# stay comparable. SSSP only emits total_time and teps (no comm/compute split),
# so the CSV schema is narrower than the BFS one.
#
# Must be run inside an allocation (e.g. after `salloc`) so that `srun` works.

set -uo pipefail

GRAPH=${1:?graph file required}
LCC_INFO=${2:?lcc info file required}
CSV=${3:?output csv required}
DELTA=${4:-10.0}    # 10.0 matches main.cpp default; appropriate for [1,100] weights

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN=$PROJECT_DIR/build/parallel_bfs
[ -x "$BIN" ] || { echo "binary not found: $BIN" >&2; exit 1; }
[ -f "$GRAPH" ]    || { echo "graph not found: $GRAPH" >&2; exit 1; }
[ -f "$LCC_INFO" ] || { echo "lcc info not found: $LCC_INFO" >&2; exit 1; }

# Parse the sources line from the LCC info file (same format strong_scaling.sh
# expects: "sources <id> <id> ..."). Random sampling without replacement so
# that we don't always benchmark the same vertex.
mapfile -t SOURCES < <(awk '/^sources/ { for (i=2; i<=NF; i++) print $i }' "$LCC_INFO")
[ ${#SOURCES[@]} -ge 2 ] || { echo "need >= 2 sources in $LCC_INFO" >&2; exit 1; }

NUM_RUNS=${NUM_RUNS:-2}

# Write CSV header if the file is new/empty. SSSP main.cpp prints only
# total_time and teps, so we don't carry comm/compute columns here.
if [ ! -s "$CSV" ]; then
    echo "algorithm,num_processes,delta,avg_total_time,avg_teps" > "$CSV"
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

    echo "=== $algo p=$nproc delta=$DELTA [$(date '+%Y-%m-%d %H:%M:%S')] ===" >&2

    # NUM_RUNS random sources without replacement from the LCC info pool.
    local picks
    mapfile -t picks < <(printf "%s\n" "${SOURCES[@]}" | shuf -n "$NUM_RUNS")

    local sum_total=0 sum_teps=0
    local n=0

    for src in "${picks[@]}"; do
        local metrics
        metrics=$(srun -n "$nproc" "$BIN" "$GRAPH" "$algo" "$src" \
                       --delta "$DELTA" --lcc-info "$LCC_INFO" --no-output 2>/dev/null \
                       | grep "^\[$algo\]" || true)
        if [ -z "$metrics" ]; then
            echo "  $algo @ p=$nproc src=$src: no metrics line (run failed?)" >&2
            continue
        fi

        local total teps
        total=$(echo "$metrics" | parse_field total_time)
        teps=$(echo  "$metrics" | parse_field teps)

        echo "  $algo @ p=$nproc src=$src: total=$total teps=$teps" >&2

        sum_total=$(awk -v a="$sum_total" -v b="$total" 'BEGIN{print a+b}')
        sum_teps=$(awk  -v a="$sum_teps"  -v b="$teps"  'BEGIN{print a+b}')
        n=$((n+1))
    done

    if [ "$n" -eq 0 ]; then
        echo "$algo @ p=$nproc: no successful runs, skipping CSV row" >&2
        return
    fi

    local avg_total avg_teps
    avg_total=$(awk -v a="$sum_total" -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_teps=$(awk  -v a="$sum_teps"  -v n="$n" 'BEGIN{printf "%.6e", a/n}')

    local row="$algo,$nproc,$DELTA,$avg_total,$avg_teps"
    echo "$row"
    echo "$row" >> "$CSV"
}

# 1D SSSP strong scaling
for P in 1 2 4 8 16 32 64 128 256; do
    run_config sssp1d "$P"
done

# 2D SSSP strong scaling (square ranks only)
for P in 1 4 16 64 256; do
    run_config sssp2d "$P"
done
