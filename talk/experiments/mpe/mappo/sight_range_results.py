"""Shared helpers for MAPPO MPE sight-range sweep results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "sight_range_results_data"

# Must match talk/experiments/mpe/mappo/sweep_sight.sh
SWEEP_SIGHT_RANGES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, -1.0]

DEFAULT_PROJECT = "talk"
DEFAULT_ENTITY = None  # resolved from wandb.Api().default_entity
DEFAULT_ALGORITHM = "mappo_mpe"
DEFAULT_ENV_NAME = "MPE_simple_spread_v3"
DEFAULT_NUM_SEEDS = 3
DEFAULT_METRIC = "return"

RUNS_CSV = "runs.csv"
SUMMARY_CSV = "summary.csv"
SUMMARY_JSON = "summary.json"


def resolve_entity(entity: str | None) -> str:
    if entity:
        return entity
    import wandb

    return wandb.Api().default_entity


def project_path(entity: str, project: str) -> str:
    return project if "/" in project else f"{entity}/{project}"


def seed_index_from_run_name(name: str) -> int | None:
    parts = name.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def normalize_sight_range(value) -> float:
    return float(value)


def results_paths(results_dir: Path) -> dict[str, Path]:
    return {
        "runs": results_dir / RUNS_CSV,
        "summary_csv": results_dir / SUMMARY_CSV,
        "summary_json": results_dir / SUMMARY_JSON,
    }


def load_runs_df(results_dir: Path) -> pd.DataFrame:
    path = results_paths(results_dir)["runs"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run download_sight_range_results.py first."
        )
    return pd.read_csv(path)


def load_summary_df(results_dir: Path) -> pd.DataFrame:
    path = results_paths(results_dir)["summary_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run download_sight_range_results.py first."
        )
    return pd.read_csv(path)


def save_summary_artifacts(summary_df: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = results_paths(results_dir)
    summary_df.to_csv(paths["summary_csv"], index=False)
    with paths["summary_json"].open("w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)


def aggregate_by_sight_range(runs_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        runs_df.groupby("sight_range", as_index=False)
        .agg(
            mean_final_return=("final_return", "mean"),
            std_final_return=("final_return", "std"),
            min_final_return=("final_return", "min"),
            max_final_return=("final_return", "max"),
            num_seeds=("final_return", "count"),
        )
        .sort_values("sight_range", key=lambda s: s.map(lambda v: (v < 0, v)))
    )
    return grouped.reset_index(drop=True)
