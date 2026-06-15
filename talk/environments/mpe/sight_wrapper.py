"""Optional sight-range masking for JaxMARL MPE observations."""

from functools import partial
from typing import Callable, Dict

import chex
import jax
import jax.numpy as jnp


def _roll_other_positions(state, num_agents: int, aidx: int) -> jnp.ndarray:
    other_pos = state.p_pos[:num_agents] - state.p_pos[aidx]
    other_pos = jnp.roll(other_pos, shift=num_agents - aidx - 1, axis=0)[: num_agents - 1]
    other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
    return other_pos


def _landmark_mask(state, num_agents: int, aidx: int, sight_range: float) -> jnp.ndarray:
    rel = state.p_pos[num_agents:] - state.p_pos[aidx]
    return jnp.sqrt(jnp.sum(jnp.square(rel), axis=1)) > sight_range


def _other_mask(state, num_agents: int, aidx: int, sight_range: float) -> jnp.ndarray:
    rel = _roll_other_positions(state, num_agents, aidx)
    return jnp.sqrt(jnp.sum(jnp.square(rel), axis=1)) > sight_range


def _mask_simple_spread(env, state, obs_dict: Dict[str, jnp.ndarray], sight_range: float):
    out = {}
    for i, a in enumerate(env.agents):
        obs = obs_dict[a]
        n_land = env.num_landmarks
        n_other = env.num_agents - 1
        lmask = _landmark_mask(state, env.num_agents, i, sight_range)
        omask = _other_mask(state, env.num_agents, i, sight_range)
        lm = jnp.repeat(lmask, 2)
        om = jnp.repeat(omask, 2)
        comm = jnp.repeat(omask, env.dim_c)
        start_land = 4
        start_other = start_land + n_land * 2
        start_comm = start_other + n_other * 2
        obs = obs.at[start_land : start_land + n_land * 2].set(
            jnp.where(lm, 0.0, obs[start_land : start_land + n_land * 2])
        )
        obs = obs.at[start_other : start_other + n_other * 2].set(
            jnp.where(om, 0.0, obs[start_other : start_other + n_other * 2])
        )
        obs = obs.at[start_comm : start_comm + n_other * env.dim_c].set(
            jnp.where(comm, 0.0, obs[start_comm : start_comm + n_other * env.dim_c])
        )
        out[a] = obs
    return out


def _mask_tag_like(env, state, obs_dict: Dict[str, jnp.ndarray], sight_range: float):
    out = {}
    n_land = env.num_landmarks
    n_other = env.num_agents - 1
    for i, a in enumerate(env.agents):
        obs = obs_dict[a]
        lmask = _landmark_mask(state, env.num_agents, i, sight_range)
        omask = _other_mask(state, env.num_agents, i, sight_range)
        lm = jnp.repeat(lmask, 2)
        om = jnp.repeat(omask, 2)

        if a.startswith("adversary"):
            start_land = 4
        else:
            start_land = max(obs.shape[0] - (n_land * 2 + n_other * 2), 0)
        start_other = start_land + n_land * 2
        obs = obs.at[start_land : start_land + n_land * 2].set(
            jnp.where(lm, 0.0, obs[start_land : start_land + n_land * 2])
        )
        obs = obs.at[start_other : start_other + n_other * 2].set(
            jnp.where(om, 0.0, obs[start_other : start_other + n_other * 2])
        )
        # FacMAC/tag adversary tails include a velocity slice for one other agent.
        if obs.shape[0] >= start_other + n_other * 2 + 2:
            tail = obs[start_other + n_other * 2 : start_other + n_other * 2 + 2]
            obs = obs.at[start_other + n_other * 2 : start_other + n_other * 2 + 2].set(
                jnp.where(omask[-1], 0.0, tail)
            )
        out[a] = obs
    return out


SIGHT_MASK_REGISTRY: Dict[str, Callable] = {
    "MPE_simple_spread_v3": _mask_simple_spread,
    "MPE_simple_tag_v3": _mask_tag_like,
    "MPE_simple_adversary_v3": _mask_tag_like,
    "MPE_simple_push_v3": _mask_tag_like,
    "MPE_simple_facmac_v1": _mask_tag_like,
    "MPE_simple_facmac_3a_v1": _mask_tag_like,
    "MPE_simple_facmac_6a_v1": _mask_tag_like,
    "MPE_simple_facmac_9a_v1": _mask_tag_like,
}


class LimitedSightWrapper:
    """Wrap an MPE env and mask out-of-range relative features."""

    def __init__(self, env, env_name: str, sight_range: float):
        self._env = env
        self.env_name = env_name
        self.sight_range = sight_range
        self._mask_fn = SIGHT_MASK_REGISTRY.get(env_name, None)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, state = self._env.reset(key)
        return self._mask_obs(state, obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state, actions):
        obs, new_state, rewards, dones, info = self._env.step(key, state, actions)
        return self._mask_obs(new_state, obs), new_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state, actions):
        obs, new_state, rewards, dones, info = self._env.step_env(key, state, actions)
        return self._mask_obs(new_state, obs), new_state, rewards, dones, info

    def _mask_obs(self, state, obs_dict: Dict[str, jnp.ndarray]):
        if self.sight_range < 0:
            return obs_dict
        if self._mask_fn is None:
            return obs_dict
        return self._mask_fn(self._env, state, obs_dict, self.sight_range)
