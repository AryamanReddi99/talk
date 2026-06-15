"""Minibatch shuffling aligned with JaxMARL MPE baselines."""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from talk.utils.typing import BoolArray, FloatArray, PRNGKeyArray


def num_team_actors(n_team: int, num_envs: int) -> int:
    return n_team * num_envs


def slot_agent_ids(n_team: int, num_envs: int) -> jnp.ndarray:
    """Agent-major actor index: slot = agent * num_envs + env."""
    return jnp.repeat(jnp.arange(n_team, dtype=jnp.int32), num_envs)


def slot_env_ids(n_team: int, num_envs: int) -> jnp.ndarray:
    """Parallel env index for each actor-major slot."""
    return jnp.tile(jnp.arange(num_envs, dtype=jnp.int32), n_team)


def hidden_to_actor_major(hidden: FloatArray, n_team: int, num_envs: int) -> FloatArray:
    """(n_team, num_envs, H) -> (num_actors, H)."""
    return hidden.reshape(num_team_actors(n_team, num_envs), -1)


def tensor_to_actor_major(
    x: FloatArray, n_team: int, num_envs: int
) -> FloatArray:
    """(T, E, n_team, *rest) or (T, E, n_team) or (T, E) -> (T, num_actors, *rest)."""
    if x.ndim == 2:
        return (
            jnp.broadcast_to(x[:, :, None], (x.shape[0], num_envs, n_team))
            .transpose(0, 2, 1)
            .reshape(x.shape[0], num_team_actors(n_team, num_envs))
        )
    if x.ndim == 3:
        return x.transpose(0, 2, 1).reshape(
            x.shape[0], num_team_actors(n_team, num_envs)
        )
    if x.ndim == 4:
        return x.transpose(0, 2, 1, 3).reshape(
            x.shape[0],
            num_team_actors(n_team, num_envs),
            x.shape[-1],
        )
    raise ValueError(f"unsupported rank for actor-major layout: {x.ndim}")


def flatten_time_actors(x: FloatArray, n_team: int, num_envs: int) -> FloatArray:
    """(T, E, n_team, *rest) -> (T * num_actors, *rest); IPPO / FF style."""
    actor_major = tensor_to_actor_major(x, n_team, num_envs)
    return actor_major.reshape(-1, *actor_major.shape[2:])


def shuffle_actor_axis(
    rng: PRNGKeyArray,
    n_team: int,
    num_envs: int,
    init_hidden: FloatArray | None,
    tensors: Tuple[FloatArray, ...],
) -> Tuple[PRNGKeyArray, FloatArray | None, Tuple[FloatArray, ...]]:
    """MAPPO-RNN style: permute over num_actors on axis 1 (after actor-major (T, A, ...))."""
    n_actors = num_team_actors(n_team, num_envs)
    rng, key = jax.random.split(rng)
    perm = jax.random.permutation(key, n_actors)
    if init_hidden is not None:
        init_hidden = jnp.take(init_hidden, perm, axis=0)
    tensors = tuple(jnp.take(t, perm, axis=1) for t in tensors)
    return rng, init_hidden, tensors


def reshape_actor_minibatches(
    x: FloatArray, num_minibatches: int, time_axis: bool = True
) -> FloatArray:
    """
    Split actor axis into minibatches.

    time_axis True: (T, num_actors, ...) -> (num_mb, T, mb_actors, ...).
    time_axis False: (num_actors, H) -> (num_mb, mb_actors, H).
    """
    if time_axis:
        return jnp.swapaxes(
            jnp.reshape(
                x,
                (x.shape[0], num_minibatches, -1, *x.shape[2:]),
            ),
            1,
            0,
        )
    return jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))


def shuffle_flat_batch(
    rng: PRNGKeyArray,
    batch_size: int,
    arrays: Tuple[FloatArray, ...],
) -> Tuple[PRNGKeyArray, Tuple[FloatArray, ...]]:
    """IPPO-FF style: permute axis 0 of (batch_size, ...)."""
    rng, key = jax.random.split(rng)
    perm = jax.random.permutation(key, batch_size)
    return rng, tuple(jnp.take(a, perm, axis=0) for a in arrays)


def reshape_flat_minibatches(
    x: FloatArray, num_minibatches: int
) -> FloatArray:
    """(batch_size, ...) -> (num_mb, mb_size, ...)."""
    return jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))


def mask_logits_actor_major(
    logits: FloatArray,
    slot_agents: jnp.ndarray,
    team_action_dim: int,
    action_dims_py: list[int],
    team_agent_indices: list[int],
) -> FloatArray:
    """Mask logits (..., batch, action_dim) using per-slot local agent id."""
    agent_dims = jnp.array(
        [action_dims_py[i] for i in team_agent_indices], dtype=jnp.int32
    )
    dims = agent_dims[slot_agents]
    if logits.ndim == 3:
        valid = jnp.arange(team_action_dim)[None, :] < dims[:, None]
    else:
        valid = jnp.arange(team_action_dim)[None, None, :] < dims[:, None, None]
    return jnp.where(valid, logits, -1e10)
