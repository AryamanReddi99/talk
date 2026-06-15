"""
Class to spawn separate wandb processes so that multiple runs
started from one python process can be logged separately.

Wandb typically only allows one process to spawn for each python
process, so this allows us to spawn multiple processes for each
wandb run, place them in a queue awaiting inputs, and log to them.

Workers use the ``spawn`` start method so they do not inherit a CUDA
context from the parent after JAX GPU initialization (fork is unsafe).
"""

import copy
import multiprocessing as mp

import numpy as np

from talk.utils import wandb_worker

# Fork after JAX has initialized GPU is unsafe; spawn gives clean worker interpreters.
_MP_CTX = mp.get_context("spawn")


def _to_loggable(obj):
    """Convert JAX/NumPy scalars to plain Python types for queue + wandb."""
    try:
        import wandb

        if isinstance(obj, (wandb.Video, wandb.Image)):
            return obj
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_loggable(v) for k, v in obj.items()}
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _plain_config(config):
    """Return a picklable copy of config for spawn workers."""
    if config is None:
        return None
    return copy.deepcopy(config)


class WandbMultiLogger:
    """
    Keeps a pair of dictionaries indexed by seed indices (0,1,etc.)
    self.processes contains references for each wandb process and
    self.queues keeps a queue for each process indexed by the same key (seed no.)
    """

    def __init__(
        self,
        project,
        group,
        job_type,
        config,
        mode,
        seed,
        num_seeds,
        save_code_path=None,
        start_workers_immediately=False,
    ):
        self._wandb_settings = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "config": _plain_config(config),
            "mode": mode,
            "seed": seed,
            "num_seeds": num_seeds,
            "save_code_path": save_code_path,
        }
        self.processes = {}
        self.queues = {}
        self._workers_started = False
        if start_workers_immediately:
            self._start_workers()

    def _start_workers(self):
        if self._workers_started:
            return
        s = self._wandb_settings
        for i in range(s["num_seeds"]):
            q = _MP_CTX.Queue()
            self.queues[i] = q
            kwargs = {
                "project": s["project"],
                "group": s["group"],
                "job_type": s["job_type"],
                "name": f"{s['seed']}_{i}_{s['group']}",
                "config": s["config"],
                "mode": s["mode"],
                "queue": q,
            }
            if i == 0 and s["save_code_path"] is not None:
                kwargs["save_code_path"] = s["save_code_path"]
            p = _MP_CTX.Process(target=wandb_worker.run, kwargs=kwargs)
            p.daemon = False
            p.start()
            self.processes[i] = p
        self._workers_started = True

    def log(self, seed, data_dict, step=None):
        if not self._workers_started:
            self._start_workers()
        data_dict = _to_loggable(data_dict)
        payload = (data_dict, step) if step is not None else data_dict
        self.queues[seed].put(payload)

    def finish(self, join_timeout=120):
        for seed in self.processes:
            self.queues[seed].put(None)
        for seed, proc in self.processes.items():
            proc.join(timeout=join_timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
