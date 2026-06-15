#!/usr/bin/env python3
"""Download MAPPO MPE sight-range sweep metrics from Weights & Biases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

from sight_range_results import (
    DEFAULT_ALGORITHM,
    DEFAULT_ENV_NAME,
    DEFAULT_METRIC,
    DEFAULT_NUM_SEEDS,
    DEFAULT_PROJECT,
    DEFAULT_RESULTS_DIR,
    SWEEP_SIGHT_RANGES,
    aggregate_by_sight_range,
    normalize_sight_range,
    project_path,
    resolve_entity,
    results_paths,
    save_summary_artifacts,
    seed_index_from_run_name,
)


def fetch_sweep_runs(
    entity: str,
    project: str,
    algorithm: str,
    env_name: str,
    metric: str,
    expected_sight_ranges: list[float],
    num_seeds: int,
) -> pd.DataFrame:
    api = wandb.Api()
    filters = {
        "config.algorithm": algorithm,
        "config.env_name": env_name,
    }
    runs = api.runs(project_path(entity, project), filters=filters)

    rows: list[dict] = []
    for run in runs:
        sight_range = run.config.get("sight_range")
        if sight_range is None:
            continue
        sight_range = normalize_sight_range(sight_range)
        if sight_range not in expected_sight_ranges:
            continue

        seed_idx = seed_index_from_run_name(run.name)
        if seed_idx is None or seed_idx >= num_seeds:
            continue

        final_return = run.summary.get(metric)
        if final_return is None:
            history = run.history(keys=[metric], pandas=True)
            if history.empty or metric not in history.columns:
                print(f"Warning: no {metric} for run {run.id} ({run.name}), skipping")
                continue
            final_return = float(history[metric].iloc[-1])
        else:
            final_return = float(final_return)

        rows.append(
            {
                "sight_range": sight_range,
                "seed_index": seed_idx,
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                "state": run.state,
                "final_return": final_return,
                "final_update_step": run.summary.get("update_step"),
                "url": run.url,
            }
        )

    if not rows:
        raise RuntimeError(
            "No matching runs found. Check entity/project filters and that the sweep finished."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["sight_range", "seed_index"]).reset_index(drop=True)
    return df


def validate_coverage(runs_df: pd.DataFrame, expected_sight_ranges: list[float], num_seeds: int) -> None:
    for sight_range in expected_sight_ranges:
        subset = runs_df[runs_df["sight_range"] == sight_range]
        if len(subset) != num_seeds:
            print(
                f"Warning: sight_range={sight_range} has {len(subset)}/{num_seeds} seeds"
            )


def download_results(
    results_dir: Path,
    entity: str | None = None,
    project: str = DEFAULT_PROJECT,
    algorithm: str = DEFAULT_ALGORITHM,
    env_name: str = DEFAULT_ENV_NAME,
    metric: str = DEFAULT_METRIC,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    sight_ranges: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity = resolve_entity(entity)
    expected = sight_ranges or SWEEP_SIGHT_RANGES

    print(f"Fetching runs from {entity}/{project} ...")
    runs_df = fetch_sweep_runs(
        entity=entity,
        project=project,
        algorithm=algorithm,
        env_name=env_name,
        metric=metric,
        expected_sight_ranges=expected,
        num_seeds=num_seeds,
    )
    validate_coverage(runs_df, expected, num_seeds)

    summary_df = aggregate_by_sight_range(runs_df)

    results_dir.mkdir(parents=True, exist_ok=True)
    paths = results_paths(results_dir)
    runs_df.to_csv(paths["runs"], index=False)
    save_summary_artifacts(summary_df, results_dir)

    print(f"Saved {len(runs_df)} runs to {paths['runs']}")
    print(f"Saved summary to {paths['summary_csv']}")
    return runs_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for downloaded CSV/JSON artifacts",
    )
    parser.add_argument("--entity", default=None, help="W&B entity (default: logged-in user)")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    runs_df, summary_df = download_results(
        results_dir=args.results_dir,
        entity=args.entity,
        project=args.project,
        algorithm=args.algorithm,
        env_name=args.env_name,
        metric=args.metric,
        num_seeds=args.num_seeds,
    )

    print("\nFinal return by sight range (mean ± std over seeds):")
    for _, row in summary_df.iterrows():
        print(
            f"  sight_range={row['sight_range']:>5}: "
            f"{row['mean_final_return']:8.3f} ± {row['std_final_return']:6.3f} "
            f"(n={int(row['num_seeds'])})"
        )
    print(f"\nTotal runs downloaded: {len(runs_df)}")


if __name__ == "__main__":
    main()
