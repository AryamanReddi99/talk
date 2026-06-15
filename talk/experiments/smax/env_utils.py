"""Helpers for constructing JaxMARL SMAX envs in talk experiments."""

import copy
from typing import Any, Dict, Optional

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.environments.smax.smax_env import Scenario


def build_smax_env_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``env_kwargs`` from Hydra config (no sight scaling)."""
    return copy.deepcopy(config.get("env_kwargs", {}))


def make_smax_env(config: Dict[str, Any], scenario: Optional[Scenario] = None):
    """
    Construct ``HeuristicEnemySMAX`` with ally-only sight scaling.

    ``sight_range_scale`` (default 1.0) scales allied observers' per-type sight
    ranges via ``AllyScaledSightSMAX``; enemy sight stays at JaxMARL defaults.
    """
    if scenario is None:
        scenario = map_name_to_scenario(config["map_name"])
    env_kwargs = build_smax_env_kwargs(config)
    scale = float(config.get("sight_range_scale", 1.0))
    return HeuristicEnemySMAX(
        scenario=scenario,
        ally_sight_range_scale=scale,
        **env_kwargs,
    )
