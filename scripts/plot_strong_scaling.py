#!/usr/bin/env python3
"""Plot strong-scaling results from a benchmark CSV.

Consumes the CSV format produced by either scripts/strong_scaling.sh (BFS) or
scripts/sssp_benchmark.sh (SSSP). Both schemas contain the columns we need:

    algorithm, num_processes, avg_teps

Plus optional avg_total_time / avg_comm_time / avg_compute_time / delta which
are ignored here.

Usage:
    scripts/plot_strong_scaling.py <csv> [--out <png>] [--title <str>]
                                         [--graph <name>]

If --graph is omitted, the basename of the CSV (minus extension) is used in
the figure title.

Example:
    scripts/plot_strong_scaling.py sssp_benchmark_results_v4.csv \\
        --out sssp_lj_scaling.png --graph com-LiveJournal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Stable colors per algorithm so 1D/2D line up across BFS and SSSP plots.
ALGO_STYLE = {
    "bfs1d":  {"color": "#1f77b4", "label": "1D BFS",  "marker": "s"},
    "bfs2d":  {"color": "#d62728", "label": "2D BFS",  "marker": "o"},
    "sssp1d": {"color": "#1f77b4", "label": "1D SSSP", "marker": "s"},
    "sssp2d": {"color": "#d62728", "label": "2D SSSP", "marker": "o"},
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"algorithm", "num_processes", "avg_teps"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV missing required columns: {sorted(missing)}")
    df = df.sort_values(["algorithm", "num_processes"]).reset_index(drop=True)
    return df


def annotate_peak(ax, sub: pd.DataFrame, color: str) -> None:
    """Mark the (rank, TEPS) at which TEPS is maximum and write the rank."""
    if sub.empty:
        return
    idx = sub["avg_teps"].idxmax()
    p_peak = int(sub.loc[idx, "num_processes"])
    teps_peak = float(sub.loc[idx, "avg_teps"])
    is_rightmost = p_peak == int(sub["num_processes"].max())
    ax.annotate(
        f"peak @ p={p_peak}",
        xy=(p_peak, teps_peak),
        xytext=(-10 if is_rightmost else 8, 10),
        textcoords="offset points",
        ha=("right" if is_rightmost else "left"),
        fontsize=9,
        color=color,
    )


def plot(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, sub in df.groupby("algorithm"):
        style = ALGO_STYLE.get(algo, {"color": "gray", "label": algo, "marker": "o"})
        ax.plot(
            sub["num_processes"],
            sub["avg_teps"],
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=2,
            label=style["label"],
        )
        annotate_peak(ax, sub, style["color"])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks (log)")
    ax.set_ylabel("TEPS (log)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Show every measured rank value on the x-axis instead of just powers of 2
    # that matplotlib happens to pick (looks more like the BFS reference plot).
    ranks = sorted(df["num_processes"].unique())
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])

    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Path to benchmark CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <csv stem>.png)")
    parser.add_argument("--graph", default=None,
                        help="Graph name to include in the title (e.g. com-LiveJournal)")
    parser.add_argument("--title", default=None,
                        help="Override the figure title entirely")
    args = parser.parse_args()

    df = load(args.csv)

    if args.title:
        title = args.title
    else:
        graph = args.graph or args.csv.stem
        # Pick a sensible "kind" from whichever algorithms appear in the file.
        algos = set(df["algorithm"].unique())
        if algos & {"sssp1d", "sssp2d"}:
            kind = "SSSP"
        elif algos & {"bfs1d", "bfs2d"}:
            kind = "BFS"
        else:
            kind = ""
        title = f"{kind} strong scaling: 1D vs 2D partitioning ({graph})".strip()

    out = args.out or args.csv.with_suffix(".png")
    plot(df, title, out)


if __name__ == "__main__":
    main()
