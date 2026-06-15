"""MPE training env factory and batching helpers (JaxMARL-style)."""

from __future__ import annotations

import functools
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp

try:
    import jaxmarl  # type: ignore[reportMissingImports]
except ModuleNotFoundError:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "JaxMARL"))
    import jaxmarl  # type: ignore[reportMissingImports]

from jaxmarl.wrappers.baselines import JaxMARLWrapper, MPELogWrapper

from talk.environments.mpe.jaxmarl_adapter import critic_state_dim
from talk.environments.mpe.sight_wrapper import LimitedSightWrapper


def ally_comm_reachability(positions: jnp.ndarray, comm_range: float) -> jnp.ndarray:
    """Agent positions (N, 2) in world units -> (N, N) bool reachability matrix."""
    n = positions.shape[0]
    if comm_range < 0:
        return jnp.ones((n, n), dtype=bool)
    dist = jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return dist <= comm_range


def mpe_agent_positions(log_state, num_agents: int) -> jnp.ndarray:
    """LogEnvState -> (E, N, 2) agent positions from MPE p_pos."""
    return log_state.env_state.p_pos[..., :num_agents, :]


def to_env_major(x: jnp.ndarray, num_envs: int, num_agents: int) -> jnp.ndarray:
    """(num_actors, ...) -> (E, N, ...) — batchify order is agent-major."""
    rest = x.shape[1:]
    return x.reshape(num_agents, num_envs, *rest).swapaxes(0, 1)


def to_actor_major(x: jnp.ndarray, num_envs: int, num_agents: int) -> jnp.ndarray:
    """(E, N, ...) -> (num_actors, ...)."""
    rest = x.shape[2:]
    return x.swapaxes(0, 1).reshape(num_agents * num_envs, *rest)


def traj_field_to_env_major(
    x: jnp.ndarray, num_envs: int, num_agents: int
) -> jnp.ndarray:
    """(T, num_actors, ...) -> (T, E, N, ...)."""
    if x.ndim == 2:
        return x.reshape(x.shape[0], num_agents, num_envs).swapaxes(1, 2)
    return x.reshape(x.shape[0], num_agents, num_envs, *x.shape[2:]).swapaxes(1, 2)


def batchify(x: dict, agent_list: Sequence[str], num_actors: int) -> jnp.ndarray:
    stacked = jnp.stack([x[a] for a in agent_list])
    return stacked.reshape((num_actors, -1))


def unbatchify(
    x: jnp.ndarray, agent_list: Sequence[str], num_envs: int, num_actors: int
) -> dict:
    x = x.reshape((num_actors, num_envs, -1))
    return {agent: x[i] for i, agent in enumerate(agent_list)}


def assert_homogeneous_discrete(env) -> None:
    """Require equal discrete obs/action dims across agents (e.g. simple_spread)."""
    agents = list(env.agents)
    action_dims = []
    obs_dims = []
    for agent in agents:
        space = env.action_space(agent)
        if not hasattr(space, "n"):
            raise ValueError(
                "mappo MPE refactor supports homogeneous discrete actions only"
            )
        action_dims.append(int(space.n))
        obs_dims.append(int(env.observation_space(agent).shape[-1]))
    if len(set(action_dims)) != 1 or len(set(obs_dims)) != 1:
        raise ValueError(
            f"heterogeneous agents not supported: action_dims={action_dims}, "
            f"obs_dims={obs_dims}"
        )


class MPEWorldStateWrapper(JaxMARLWrapper):
    """Attach Talk-style centralized critic world state to dict observations."""

    def __init__(self, env):
        super().__init__(env)
        self._base = self._unwrapped_env()
        self._obs_dims_py = [
            int(self._base.observation_space(a).shape[-1]) for a in self._base.agents
        ]
        self._max_obs_dim = int(max(self._obs_dims_py))
        self._num_agents = int(self._base.num_agents)
        self._world_state_size = critic_state_dim(self._num_agents, self._max_obs_dim)

    def _unwrapped_env(self):
        env = self._env
        while hasattr(env, "_env"):
            env = env._env
        return env

    def world_state_size(self) -> int:
        return int(self._world_state_size)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _world_state_from_state(self, state) -> jnp.ndarray:
        obs_dict = self._base.get_obs(state)
        stacked = jnp.zeros((self._num_agents, self._max_obs_dim), dtype=jnp.float32)
        for i, agent in enumerate(self._base.agents):
            dim = self._obs_dims_py[i]
            stacked = stacked.at[i, :dim].set(obs_dict[agent].astype(jnp.float32))
        joint = stacked.reshape(self._num_agents * self._max_obs_dim)
        one_hot = jnp.eye(self._num_agents, dtype=jnp.float32)
        joint_b = jnp.broadcast_to(joint[None, :], (self._num_agents, joint.shape[0]))
        return jnp.concatenate([joint_b, one_hot], axis=-1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, state = self._env.reset(key)
        obs["world_state"] = self._world_state_from_state(state)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action):
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self._world_state_from_state(state)
        return obs, state, reward, done, info


def _env_kwargs_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    env_kwargs = dict(config.get("env_kwargs", {}) or {})
    num_agents = config.get("num_agents")
    env_name = config["env_name"]
    if num_agents is not None:
        env_kwargs["num_agents"] = int(num_agents)
        if env_name == "MPE_simple_spread_v3" and "num_landmarks" not in env_kwargs:
            env_kwargs["num_landmarks"] = int(num_agents)
    env_kwargs.setdefault("action_type", "Discrete")
    return env_kwargs


def make_mpe_train_env(config: Dict[str, Any]):
    """jaxmarl.make -> optional LimitedSightWrapper -> world state -> MPELogWrapper."""
    env_name = config["env_name"]
    env_kwargs = _env_kwargs_from_config(config)
    env = jaxmarl.make(env_name, **env_kwargs)
    assert_homogeneous_discrete(env)

    sight_range = float(config.get("sight_range", -1.0))
    if sight_range >= 0:
        env = LimitedSightWrapper(env, env_name=env_name, sight_range=sight_range)

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)
    return env
