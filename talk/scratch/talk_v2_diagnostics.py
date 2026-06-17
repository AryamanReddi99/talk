"""Pull diagnostic tail-means for talk_gumbel_v2 _new_mpe runs to test hypotheses.

For each talk run, in a single history() call, grab the last-10% mean of the
codebook/attention diagnostics and the train/test returns, then aggregate by
(sight_range, codebook_size, aux_coef).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wandb

CUSTOM_NAME = "_new_mpe"
TALK_ALGO = "mappo_talk_v2"
TAIL_FRAC = 0.1
OUT_DIR = Path(__file__).resolve().parent

KEYS = [
    "returns",
    "test_returns",
    "aux_loss",
    "raw_align_l2",
    "token_entropy",
    "attention_entropy",
    "self_attention",
    "comm_norm",
    "entropy",
]


def main() -> None:
    api = wandb.Api()
    ent = api.default_entity
    runs = api.runs(
        f"{ent}/talk",
        filters={"config.custom_name": CUSTOM_NAME, "config.algorithm": TALK_ALGO},
        per_page=500,
    )

    rows = []
    for run in runs:
        cfg = run.config
        try:
            hist = run.history(keys=KEYS, samples=5000)
        except Exception:
            hist = None
        rec = {
            "sight_range": float(cfg.get("sight_range", -99)),
            "codebook_size": int(cfg.get("codebook_size", -1)),
            "aux_coef": float(cfg.get("aux_coef", -1)),
            "run_id": run.id,
        }
        for k in KEYS:
            v = np.nan
            if hist is not None and k in hist:
                vals = pd.to_numeric(hist[k], errors="coerce").dropna().to_numpy()
                if vals.size:
                    n = max(1, int(TAIL_FRAC * vals.size))
                    v = float(np.mean(vals[-n:]))
            rec[k] = v
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "talk_v2_diagnostics_runs.csv", index=False)

    agg = (
        df.groupby(["sight_range", "codebook_size", "aux_coef"], as_index=False)
        .agg(
            n=("returns", "count"),
            returns=("returns", "mean"),
            test_returns=("test_returns", "mean"),
            aux_loss=("aux_loss", "mean"),
            raw_align_l2=("raw_align_l2", "mean"),
            token_entropy=("token_entropy", "mean"),
            attn_entropy=("attention_entropy", "mean"),
            self_attn=("self_attention", "mean"),
            comm_norm=("comm_norm", "mean"),
        )
        .sort_values(["sight_range", "codebook_size", "aux_coef"])
        .reset_index(drop=True)
    )
    agg["train_test_gap"] = agg["returns"] - agg["test_returns"]
    agg.to_csv(OUT_DIR / "talk_v2_diagnostics_summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
