"""Probe message-health metrics in existing talk coef=0 runs."""
import numpy as np
import wandb

api = wandb.Api()
ENT = api.default_entity
runs = api.runs(f"{ENT}/talk", per_page=500)

keys = ["returns", "msg_len", "expected_msg_len", "silence_rate",
        "token_entropy", "comm_ctx_norm", "actor_grad_norm", "critic_grad_norm",
        "entropy", "gumbel_tau"]

shown = 0
for run in runs:
    cfg = run.config
    if cfg.get("custom_name") != "_new_mpe" or cfg.get("algorithm") != "mappo_talk":
        continue
    if abs(float(cfg.get("len_aux_coef", -1))) > 1e-9:
        continue  # only coef=0
    sight = float(cfg.get("sight_range", -99))
    avail = set(run.summary.keys())
    present = [k for k in keys if k in avail]
    print(f"\nrun {run.name} sight={sight} talk_config={cfg.get('talk_config','')!r}")
    print("  metric keys present:", present)
    if present:
        hist = run.history(keys=[k for k in present], samples=500)
        for k in present:
            if k in hist:
                v = hist[k].dropna().to_numpy()
                if v.size:
                    n = max(1, int(0.1 * v.size))
                    print(f"    {k:18} final~{np.mean(v[-n:]):.4f}")
    shown += 1
    if shown >= 4:
        break
