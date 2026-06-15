#!/usr/bin/env python3
"""Plot final MAPPO return vs sight range (mean over 3 seeds)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sight_range_results import DEFAULT_RESULTS_DIR, load_summary_df


def plot_final_return(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str,
    show: bool = False,
) -> None:
    df = summary_df.sort_values(
        "sight_range", key=lambda s: s.map(lambda v: (v < 0, v))
    ).reset_index(drop=True)

    x = df["sight_range"].to_numpy()
    y = df["mean_final_return"].to_numpy()
    yerr = df["std_final_return"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label="mean ± std (3 seeds)",
    )
    ax.set_xlabel("Sight range")
    ax.set_ylabel("Final return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing summary.csv from download script",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <results-dir>/final_return_vs_sight_range.png)",
    )
    parser.add_argument(
        "--title",
        default="MAPPO MPE simple_spread: final return vs sight range",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    summary_df = load_summary_df(args.results_dir)
    output_path = args.output or (args.results_dir / "final_return_vs_sight_range.png")
    plot_final_return(summary_df, output_path, title=args.title, show=args.show)


if __name__ == "__main__":
    main()
