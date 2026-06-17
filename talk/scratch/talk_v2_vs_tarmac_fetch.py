"""Fetch + aggregate _new_mpe runs to compare talk_gumbel_v2 vs tarmac baseline.

Downloads all wandb runs with config.custom_name == "_new_mpe" for the two
algorithms (mappo_talk_v2, mappo_tarmac_mpe), computes final-performance
metrics (mean of last 10% of the `returns` / `test_returns` history), and
aggregates over seeds by (algorithm, sight_range, codebook_size, aux_coef).

Outputs:
  talk/scratch/talk_v2_vs_tarmac_runs.csv     (one row per wandb run/seed)
  talk/scratch/talk_v2_vs_tarmac_summary.csv  (one row per config group)
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

CUSTOM_NAME = "_new_mpe"
TALK_ALGO = "mappo_talk_v2"
TARMAC_ALGO = "mappo_tarmac_mpe"
ALGOS = (TALK_ALGO, TARMAC_ALGO)
TAIL_FRAC = 0.1
OUT_DIR = Path(__file__).resolve().parent


def tail_mean(run, key: str):
    try:
        hist = run.history(keys=[key], samples=5000)
    except Exception:
        return None
    if hist is None or key not in hist or len(hist) == 0:
        return None
    vals = pd.to_numeric(hist[key], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return None
    n = max(1, int(TAIL_FRAC * vals.size))
    return float(np.mean(vals[-n:])), float(np.max(vals)), int(vals.size)


def main() -> None:
    api = wandb.Api()
    ent = api.default_entity
    print(f"Fetching {ent}/talk runs with custom_name={CUSTOM_NAME!r} ...")
    runs = api.runs(
        f"{ent}/talk",
        filters={"config.custom_name": CUSTOM_NAME},
        per_page=500,
    )

    rows: list[dict] = []
    for run in runs:
        cfg = run.config
        algo = cfg.get("algorithm", "?")
        if algo not in ALGOS:
            continue

        ret = tail_mean(run, "returns")
        if ret is None:
            continue
        final_ret, max_ret, n_pts = ret

        test = tail_mean(run, "test_returns")
        final_test = test[0] if test is not None else np.nan

        rows.append(
            {
                "algo": algo,
                "sight_range": float(cfg.get("sight_range", -99)),
                "codebook_size": int(cfg.get("codebook_size", -1)) if algo == TALK_ALGO else -1,
                "aux_coef": float(cfg.get("aux_coef", -1)) if algo == TALK_ALGO else -1.0,
                "comm_range": float(cfg.get("comm_range", -99)),
                "talk_config": str(cfg.get("talk_config", "")) if algo == TALK_ALGO else "",
                "vocab_dim": int(cfg.get("vocab_dim", -1)) if algo == TALK_ALGO else -1,
                "sig_dim": int(cfg.get("sig_dim", -1)),
                "gumbel_tau": float(cfg.get("gumbel_tau", -1)) if algo == TALK_ALGO else -1.0,
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                "state": run.state,
                "final_return": final_ret,
                "max_return": max_ret,
                "final_test_return": final_test,
                "n_points": n_pts,
                "url": run.url,
            }
        )

    if not rows:
        raise RuntimeError("No matching runs found.")

    runs_df = pd.DataFrame(rows)
    runs_df = runs_df.sort_values(
        ["algo", "sight_range", "codebook_size", "aux_coef"]
    ).reset_index(drop=True)
    runs_csv = OUT_DIR / "talk_v2_vs_tarmac_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    group_keys = ["algo", "sight_range", "codebook_size", "aux_coef"]
    summary = (
        runs_df.groupby(group_keys, as_index=False)
        .agg(
            n_seeds=("final_return", "count"),
            mean_return=("final_return", "mean"),
            std_return=("final_return", "std"),
            mean_max_return=("max_return", "mean"),
            mean_test_return=("final_test_return", "mean"),
        )
        .sort_values(["algo", "sight_range", "codebook_size", "aux_coef"])
        .reset_index(drop=True)
    )
    summary_csv = OUT_DIR / "talk_v2_vs_tarmac_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"\nMatched {len(runs_df)} runs across {len(summary)} config groups")
    print(f"Saved {runs_csv}")
    print(f"Saved {summary_csv}\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
