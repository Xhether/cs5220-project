#!/usr/bin/env bash
# Weak scaling experiment: runs 1D and 2D BFS at increasing rank counts and
# graph sizes, averaging timing over NUM_RUNS source vertices per config.
#
# Run from inside an salloc with enough nodes:
#   salloc --nodes 4 --qos interactive --time 02:00:00 --constraint cpu --account=m4341
#   bash scripts/weak_scaling.sh
#
# Output: weak_scaling_results.csv (one averaged row per (algo, ranks, scale))

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BINARY="$PROJECT_DIR/build/parallel_bfs"
GRAPH_DIR="${GRAPH_DIR:-$SCRATCH/graphs}"   # graph files live in $SCRATCH/graphs
LCC_DIR="${LCC_DIR:-$PROJECT_DIR}"          # *_lcc.txt files live in the repo root
OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_DIR/weak_scaling_results.csv}"
NUM_RUNS=2   # how many source vertices to average per config

# Weak scaling configs: "<ranks> <scale>"  (m_per_rank = 16 * 2^scale / ranks)
CONFIGS=(
    "1 18"
    "4 20"
    "16 22"
    "64 24"
)

[ -x "$BINARY" ] || { echo "binary not found: $BINARY" >&2; exit 1; }

# ── CSV header ───────────────────────────────────────────────────────────────
echo "algo,ranks,scale,n_lcc,m_lcc,avg_total_time,avg_comm_time,avg_compute_time,avg_teps" > "$OUTPUT_CSV"

# ── Helpers ──────────────────────────────────────────────────────────────────
# Extract `key=value` token from a metrics line on stdin.
parse_field() {
    awk -v key="$1" '{
        for (i=1; i<=NF; i++) {
            split($i, kv, "=")
            if (kv[1] == key) { print kv[2]; exit }
        }
    }'
}

# Square check for 2D BFS rank guard.
is_square() {
    local r
    r=$(awk -v n="$1" 'BEGIN{printf "%d", sqrt(n)+0.5}')
    [ $((r * r)) -eq "$1" ]
}

# Run one (algo, ranks, scale) cell averaged over NUM_RUNS sources.
run_cell() {
    local algo=$1 ranks=$2 scale=$3 graph=$4 lcc=$5
    shift 5
    local sources=("$@")

    echo "  → $algo ranks=$ranks scale=$scale  (averaging $NUM_RUNS runs)"

    local sum_total=0 sum_comm=0 sum_compute=0 sum_teps=0 n=0
    local n_lcc="" m_lcc=""

    for src in "${sources[@]}"; do
        local metrics
        metrics=$(srun -n "$ranks" "$BINARY" "$graph" "$algo" "$src" \
                       --lcc-info "$lcc" --no-output 2>/dev/null \
                       | grep "^\[$algo\]" || true)
        if [ -z "$metrics" ]; then
            echo "      src=$src: no metrics line (run failed?)"
            continue
        fi

        local total comm cmpu teps
        total=$(echo "$metrics" | parse_field total_time)
        comm=$(echo  "$metrics" | parse_field comm_time)
        cmpu=$(echo  "$metrics" | parse_field compute_time)
        teps=$(echo  "$metrics" | parse_field teps)
        n_lcc=$(echo "$metrics" | parse_field n_lcc)
        m_lcc=$(echo "$metrics" | parse_field m_lcc)

        printf "      src=%-10s total=%-10s comm=%-10s compute=%-10s teps=%s\n" \
               "$src" "$total" "$comm" "$cmpu" "$teps"

        sum_total=$(awk   -v a="$sum_total"   -v b="$total" 'BEGIN{print a+b}')
        sum_comm=$(awk    -v a="$sum_comm"    -v b="$comm"  'BEGIN{print a+b}')
        sum_compute=$(awk -v a="$sum_compute" -v b="$cmpu"  'BEGIN{print a+b}')
        sum_teps=$(awk    -v a="$sum_teps"    -v b="$teps"  'BEGIN{print a+b}')
        n=$((n+1))
    done

    if [ "$n" -eq 0 ]; then
        echo "      no successful runs, skipping CSV row"
        return
    fi

    local avg_total avg_comm avg_compute avg_teps
    avg_total=$(awk   -v a="$sum_total"   -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_comm=$(awk    -v a="$sum_comm"    -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_compute=$(awk -v a="$sum_compute" -v n="$n" 'BEGIN{printf "%.6e", a/n}')
    avg_teps=$(awk    -v a="$sum_teps"    -v n="$n" 'BEGIN{printf "%.6e", a/n}')

    local row="$algo,$ranks,$scale,$n_lcc,$m_lcc,$avg_total,$avg_comm,$avg_compute,$avg_teps"
    echo "      avg: total=$avg_total comm=$avg_comm compute=$avg_compute teps=$avg_teps"
    echo "$row" >> "$OUTPUT_CSV"
}

# ── Main loop ────────────────────────────────────────────────────────────────
echo "Weak scaling sweep starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  binary:    $BINARY"
echo "  graphs:    $GRAPH_DIR"
echo "  lcc info:  $LCC_DIR"
echo "  output:    $OUTPUT_CSV"
echo "  runs/cell: $NUM_RUNS"

for cfg in "${CONFIGS[@]}"; do
    read -r RANKS SCALE <<< "$cfg"
    GRAPH="$GRAPH_DIR/graph_s${SCALE}.txt"
    LCC="$LCC_DIR/graph_s${SCALE}_lcc.txt"

    echo ""
    echo "=========================================="
    echo " ranks=$RANKS scale=$SCALE"
    echo "   graph=$GRAPH"
    echo "   lcc=$LCC"
    echo "=========================================="

    if [[ ! -f "$GRAPH" ]]; then
        echo "  WARN: missing $GRAPH, skipping"
        continue
    fi
    if [[ ! -f "$LCC" ]]; then
        echo "  WARN: missing $LCC, skipping"
        continue
    fi

    # Pick NUM_RUNS random sources from the LCC info file.
    mapfile -t ALL_SOURCES < <(awk '/^sources/ { for (i=2; i<=NF; i++) print $i }' "$LCC")
    if [ "${#ALL_SOURCES[@]}" -lt "$NUM_RUNS" ]; then
        echo "  WARN: only ${#ALL_SOURCES[@]} sources in $LCC (need $NUM_RUNS), skipping"
        continue
    fi
    mapfile -t PICKS < <(printf "%s\n" "${ALL_SOURCES[@]}" | shuf -n "$NUM_RUNS")
    echo "  sources: ${PICKS[*]}"

    # 1D BFS
    run_cell "bfs1d" "$RANKS" "$SCALE" "$GRAPH" "$LCC" "${PICKS[@]}"

    # 2D BFS (requires square ranks)
    if is_square "$RANKS"; then
        run_cell "bfs2d" "$RANKS" "$SCALE" "$GRAPH" "$LCC" "${PICKS[@]}"
    else
        echo "  → bfs2d skipped: ranks=$RANKS is not a perfect square"
    fi
done

echo ""
echo "Done at $(date '+%Y-%m-%d %H:%M:%S'). Results in $OUTPUT_CSV"
echo "Rows: $(($(wc -l < "$OUTPUT_CSV") - 1))"
