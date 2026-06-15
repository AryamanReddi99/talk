"""Monitor running/finished talk runs by talk_config tag. Usage: python -m talk.scratch.wandb_monitor tag1 tag2 ..."""
import sys
import collections
import numpy as np
import wandb

api = wandb.Api()
ENT = api.default_entity
tags = sys.argv[1:] if len(sys.argv) > 1 else None

runs = api.runs(f"{ENT}/talk", per_page=500,
                filters={"config.custom_name": "_new_mpe", "config.algorithm": "mappo_talk"})

groups = collections.defaultdict(list)
extra = collections.defaultdict(lambda: collections.defaultdict(list))
KEYS = ["returns", "msg_len", "silence_rate", "comm_ctx_norm", "token_entropy", "_step"]
for run in runs:
    cfg = run.config
    tc = str(cfg.get("talk_config", ""))
    if tags and tc not in tags:
        continue
    sight = float(cfg.get("sight_range", -99))
    try:
        hist = run.history(keys=[k for k in KEYS if k != "_step"], samples=1000)
    except Exception:
        continue
    if hist is None or "returns" not in hist:
        continue
    ret = hist["returns"].dropna().to_numpy()
    if ret.size == 0:
        continue
    n = max(1, int(0.1 * ret.size))
    key = (tc, sight)
    groups[key].append(float(np.mean(ret[-n:])))
    for k in ["msg_len", "silence_rate", "comm_ctx_norm", "token_entropy"]:
        if k in hist:
            v = hist[k].dropna().to_numpy()
            if v.size:
                extra[key][k].append(float(np.mean(v[-n:])))

print(f"{'talk_config':22} {'sight':>6} {'n':>2} {'ret':>9} {'std':>6} {'msglen':>7} {'silence':>8} {'ctxnorm':>8} {'tokent':>7}")
for key in sorted(groups):
    tc, sight = key
    vals = np.array(groups[key])
    e = extra[key]
    def m(k):
        return np.mean(e[k]) if e.get(k) else float("nan")
    print(f"{tc:22} {sight:6.2f} {len(vals):2d} {vals.mean():9.3f} {vals.std():6.2f} "
          f"{m('msg_len'):7.2f} {m('silence_rate'):8.3f} {m('comm_ctx_norm'):8.3f} {m('token_entropy'):7.3f}")
