"""Fetch + aggregate _new_mpe runs for talk vs tarmac comparison."""
import collections
import json
import numpy as np
import wandb

api = wandb.Api()
ENT = api.default_entity
runs = api.runs(f"{ENT}/talk", per_page=500)

groups = collections.defaultdict(list)  # key -> list of final returns


def final_return(run):
    try:
        hist = run.history(keys=["returns"], samples=2000)
    except Exception:
        return None
    if hist is None or "returns" not in hist or len(hist) == 0:
        return None
    vals = hist["returns"].dropna().to_numpy()
    if vals.size == 0:
        return None
    n = max(1, int(0.1 * vals.size))
    return float(np.mean(vals[-n:]))


count = 0
for run in runs:
    cfg = run.config
    if cfg.get("custom_name") != "_new_mpe":
        continue
    algo = cfg.get("algorithm", "?")
    if algo not in ("mappo_talk", "mappo_tarmac_mpe"):
        continue
    fr = final_return(run)
    if fr is None:
        continue
    key = (
        algo,
        float(cfg.get("sight_range", -99)),
        float(cfg.get("comm_range", -99)),
        float(cfg.get("len_aux_coef", -1)) if algo == "mappo_talk" else -1,
        str(cfg.get("talk_config", "")) if algo == "mappo_talk" else "",
    )
    groups[key].append(fr)
    count += 1

print(f"matched {count} runs across {len(groups)} config groups\n")
print(f"{'algo':18} {'sight':>6} {'comm':>6} {'lenc':>5} {'talk_config':16} {'n':>2} {'mean_ret':>9} {'std':>7}")
rows = []
for key in sorted(groups, key=lambda k: (k[0], k[1], k[3], k[4])):
    algo, sight, comm, lenc, tc = key
    vals = np.array(groups[key])
    print(f"{algo:18} {sight:6.2f} {comm:6.1f} {lenc:5.2f} {tc:16} {len(vals):2d} {vals.mean():9.3f} {vals.std():7.3f}")
    rows.append({"algo": algo, "sight": sight, "comm": comm, "len_aux_coef": lenc,
                 "talk_config": tc, "n": len(vals), "mean": float(vals.mean()),
                 "std": float(vals.std()), "vals": [float(v) for v in vals]})

with open("talk/scratch/wandb_summary.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nsaved talk/scratch/wandb_summary.json")
