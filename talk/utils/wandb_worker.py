"""Minimal W&B logging process (no JAX imports — safe for spawn after GPU training)."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import wandb  # noqa: E402


def run(project, group, job_type, name, config, mode, queue, save_code_path=None):
    wandb.init(
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        config=config,
        mode=mode,
    )
    try:
        while True:
            data = queue.get()
            if data is None:
                break
            if isinstance(data, tuple) and len(data) == 2:
                payload, step = data
                wandb.log(payload, step=step)
            else:
                wandb.log(data)
    finally:
        wandb.finish()
