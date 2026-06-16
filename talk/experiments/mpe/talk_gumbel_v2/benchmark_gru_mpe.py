"""Benchmark MPE/SMAX MAPPO vs MAPPO-GRU to diagnose slowdown."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

# Repo root: talk/experiments/mpe/mappo_gru -> 4 parents
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class BenchResult:
    name: str
    compile_s: float
    run_s: float
    num_updates: int
    grad_steps_per_update: int

    @property
    def total_s(self) -> float:
        return self.compile_s + self.run_s

    @property
    def s_per_update(self) -> float:
        return self.run_s / max(self.num_updates, 1)


def _load_yaml(rel_path: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(REPO_ROOT / rel_path))


def _prepare_mpe_config(
    base_rel: str, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    cfg = _load_yaml(base_rel)
    cfg.update(
        {
            "total_timesteps": 655360,
            "num_seeds": 1,
            "use_wandb": False,
            "log_rollout_videos": False,
        }
    )
    if overrides:
        cfg.update(overrides)
    cfg["num_updates"] = (
        cfg["total_timesteps"]
        // cfg["num_envs"]
        // cfg["num_steps_per_env_per_update"]
    )
    return cfg


def _prepare_smax_config(
    base_rel: str, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    cfg = _load_yaml(base_rel)
    cfg.update(
        {
            "total_timesteps": 655360,
            "num_seeds": 1,
            "use_wandb": False,
        }
    )
    if overrides:
        cfg.update(overrides)
    num_envs = int(cfg["num_envs"])
    num_steps = int(cfg["num_steps_per_env_per_update"])
    cfg["num_updates"] = int(cfg["total_timesteps"]) // num_steps // num_envs
    return cfg


def _time_train(
    name: str,
    make_train_fn: Callable[[Dict[str, Any]], Callable],
    config: Dict[str, Any],
) -> BenchResult:
    print(f"\n=== {name} ===")
    print(
        f"  updates={config.get('num_updates', 0)} "
        f"epochs={config['num_epochs']} minibatches={config['num_minibatches']} "
        f"grad_steps/update={config['num_epochs'] * config['num_minibatches']}"
    )

    rng = jax.random.PRNGKey(int(config.get("seed", 0)))
    rng_seeds = jax.random.split(rng, int(config["num_seeds"]))
    exp_ids = jnp.arange(int(config["num_seeds"]))

    t0 = time.perf_counter()
    train_fn = jax.jit(jax.vmap(make_train_fn(config)))
    t_compile_start = time.perf_counter()
    print("  compiling...")
    out = train_fn(rng_seeds, exp_ids)
    jax.block_until_ready(out)
    t1 = time.perf_counter()

    compile_s = t_compile_start - t0
    run_s = t1 - t_compile_start
    num_updates = int(config.get("num_updates", 0))
    result = BenchResult(
        name=name,
        compile_s=compile_s,
        run_s=run_s,
        num_updates=num_updates,
        grad_steps_per_update=config["num_epochs"] * config["num_minibatches"],
    )
    print(
        f"  compile={compile_s:.2f}s run={run_s:.2f}s "
        f"total={result.total_s:.2f}s ({result.s_per_update:.3f}s/update)"
    )
    return result


def run_benchmark_matrix(include_smax: bool = True) -> list[BenchResult]:
    from talk.experiments.mpe.mappo.mappo import make_train as make_mpe_mappo
    from talk.experiments.mpe.mappo_gru.mappo_gru import make_train as make_mpe_gru

    cases: list[tuple[str, Callable, Dict[str, Any]]] = [
        (
            "T0_mpe_mappo",
            make_mpe_mappo,
            _prepare_mpe_config("talk/experiments/mpe/mappo/config_mappo.yaml"),
        ),
        (
            "T1_mpe_gru_default",
            make_mpe_gru,
            _prepare_mpe_config("talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml"),
        ),
        (
            "T2_mpe_gru_smax_ppo",
            make_mpe_gru,
            _prepare_mpe_config(
                "talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml",
                {"num_epochs": 2, "num_minibatches": 2},
            ),
        ),
        (
            "T3_mpe_gru_jaxmarl_ppo",
            make_mpe_gru,
            _prepare_mpe_config(
                "talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml",
                {"num_epochs": 4, "num_minibatches": 4},
            ),
        ),
    ]

    if include_smax:
        from talk.experiments.smax.mappo.mappo import make_train as make_smax_mappo
        from talk.experiments.smax.mappo_gru.mappo_gru import (
            make_train as make_smax_gru,
        )

        cases.extend(
            [
                (
                    "T4_smax_mappo",
                    make_smax_mappo,
                    _prepare_smax_config(
                        "talk/experiments/smax/mappo/config_mappo.yaml",
                        {"map_name": "3s_vs_5z"},
                    ),
                ),
                (
                    "T5_smax_gru",
                    make_smax_gru,
                    _prepare_smax_config(
                        "talk/experiments/smax/mappo_gru/config_mappo_gru.yaml",
                        {"map_name": "8m"},
                    ),
                ),
            ]
        )

    results = []
    for name, make_train_fn, cfg in cases:
        # Recompute derived fields after overrides
        results.append(_time_train(name, make_train_fn, cfg))
    return results


def _print_summary(results: list[BenchResult]) -> None:
    by_name = {r.name: r for r in results}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'name':<28} {'compile':>8} {'run':>8} {'s/up':>8} {'grad/up':>8}")
    for r in results:
        print(
            f"{r.name:<28} {r.compile_s:8.2f} {r.run_s:8.2f} "
            f"{r.s_per_update:8.3f} {r.grad_steps_per_update:8d}"
        )

    if "T0_mpe_mappo" in by_name and "T1_mpe_gru_default" in by_name:
        t0, t1 = by_name["T0_mpe_mappo"], by_name["T1_mpe_gru_default"]
        print(f"\nMPE GRU / MPE MLP run ratio: {t1.run_s / t0.run_s:.2f}x")

    if "T1_mpe_gru_default" in by_name and "T2_mpe_gru_smax_ppo" in by_name:
        t1, t2 = by_name["T1_mpe_gru_default"], by_name["T2_mpe_gru_smax_ppo"]
        grad_ratio = t1.grad_steps_per_update / t2.grad_steps_per_update
        time_ratio = t1.run_s / t2.run_s
        print(
            f"MPE GRU default / SMAX-PPO-budget GRU: {time_ratio:.2f}x "
            f"(grad steps ratio {grad_ratio:.1f}x, time/grad-step ratio "
            f"{time_ratio / grad_ratio:.2f}x)"
        )

    if "T4_smax_mappo" in by_name and "T5_smax_gru" in by_name:
        t4, t5 = by_name["T4_smax_mappo"], by_name["T5_smax_gru"]
        print(f"SMAX GRU / SMAX MLP run ratio: {t5.run_s / t4.run_s:.2f}x")


def timing_split_analysis(num_updates: int = 10) -> None:
    """Estimate rollout vs PPO time from ablation configs (profiler-free)."""
    from talk.experiments.mpe.mappo.mappo import make_train as make_mpe_mappo
    from talk.experiments.mpe.mappo_gru.mappo_gru import make_train as make_mpe_gru

    ts = num_updates * 128 * 64
    base = {
        "total_timesteps": ts,
        "num_seeds": 1,
        "use_wandb": False,
        "log_rollout_videos": False,
    }

    def _run(name: str, make_fn, cfg_path: str, ppo_overrides: dict) -> float:
        cfg = _prepare_mpe_config(cfg_path, {**base, **ppo_overrides})
        cfg["num_updates"] = num_updates
        return _time_train(name, make_fn, cfg).run_s

    mlp_s = _run(
        "split_mlp",
        make_mpe_mappo,
        "talk/experiments/mpe/mappo/config_mappo.yaml",
        {"num_epochs": 10, "num_minibatches": 8},
    )
    gru_full_s = _run(
        "split_gru_80",
        make_mpe_gru,
        "talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml",
        {"num_epochs": 10, "num_minibatches": 8},
    )
    gru_low_s = _run(
        "split_gru_4",
        make_mpe_gru,
        "talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml",
        {"num_epochs": 2, "num_minibatches": 2},
    )

    # gru_low = rollout_gru + 4 * ppo_step
    # gru_full = rollout_gru + 80 * ppo_step
    ppo_step = (gru_full_s - gru_low_s) / (80 - 4)
    rollout_gru_est = gru_low_s - 4 * ppo_step
    ppo_full_est = 80 * ppo_step

    print("\n" + "=" * 60)
    print(f"TIMING SPLIT ({num_updates} updates, profiler-free estimate)")
    print("=" * 60)
    print(f"  MLP total (rollout+80 FF-PPO):     {mlp_s:.2f}s")
    print(f"  GRU total (rollout+80 RNN-PPO):    {gru_full_s:.2f}s")
    print(f"  GRU total (rollout+4 RNN-PPO):     {gru_low_s:.2f}s")
    print(f"  Est. GRU rollout+GAE per update:   {rollout_gru_est / num_updates:.3f}s")
    print(f"  Est. RNN-PPO per grad step:        {ppo_step / num_updates:.4f}s")
    print(
        f"  Est. fraction in 80-step PPO:      "
        f"{ppo_full_est / gru_full_s * 100:.1f}%"
    )
    print(
        f"  Est. fraction in rollout+GAE:      "
        f"{rollout_gru_est / gru_full_s * 100:.1f}%"
    )


def profile_one_mpe_gru_update(trace_dir: str | None = None) -> None:
    """Profile a single MPE GRU update to see rollout vs PPO cost."""
    from talk.experiments.mpe.mappo_gru.mappo_gru import make_train

    cfg = _prepare_mpe_config(
        "talk/experiments/mpe/mappo_gru/config_mappo_gru.yaml",
        {"total_timesteps": 128 * 64},
    )
    cfg["num_updates"] = 1

    rng = jax.random.PRNGKey(0)
    train_fn = jax.jit(jax.vmap(make_train(cfg)))

    # Warmup compile
    _ = train_fn(jax.random.split(rng, 1), jnp.array([0]))
    jax.block_until_ready(_)

    if trace_dir is None:
        trace_dir = str(REPO_ROOT / "talk/experiments/mpe/mappo_gru/profile_trace")

    print(f"\nProfiling 1 update -> {trace_dir}")
    jax.profiler.start_trace(trace_dir)
    out = train_fn(jax.random.split(rng, 1), jnp.array([0]))
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    print("Profile trace written.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MAPPO vs MAPPO-GRU")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run JAX profiler for 1 MPE GRU update (may segfault on some setups)",
    )
    parser.add_argument(
        "--timing-split",
        action="store_true",
        help="Estimate rollout vs PPO time via ablation (no profiler)",
    )
    parser.add_argument(
        "--split-updates",
        type=int,
        default=10,
        help="Number of updates for --timing-split",
    )
    parser.add_argument(
        "--no-smax",
        action="store_true",
        help="Skip SMAX benchmarks (T4/T5)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Output directory for JAX profiler trace",
    )
    args = parser.parse_args()

    if args.timing_split:
        timing_split_analysis(num_updates=args.split_updates)
        return

    if args.profile:
        profile_one_mpe_gru_update(args.trace_dir)
        return

    results = run_benchmark_matrix(include_smax=not args.no_smax)
    _print_summary(results)


if __name__ == "__main__":
    main()
