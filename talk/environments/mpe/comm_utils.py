"""Distance-based neighbor messaging for MPE comm-range policies."""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from talk.environments.mpe.jaxmarl_adapter import TeamSpec
from talk.utils.typing import FloatArray


def max_neighbor_slots(team_specs: Sequence[TeamSpec]) -> int:
    """Upper bound on in-range neighbors per agent (at least one slot)."""
    return max(max(len(t.agent_indices) - 1, 1) for t in team_specs)


def gather_neighbors_by_comm_range(
    all_msgs: FloatArray,
    team_positions: FloatArray,
    comm_range: float,
    max_neighbor_slots: int,
) -> Tuple[FloatArray, FloatArray]:
    """
    Gather neighbor messages by Euclidean distance within ``comm_range``.

    ``comm_range < 0`` means unlimited range (all teammates except self).

    Args:
        all_msgs: (n_team, num_envs, msg_dim)
        team_positions: (num_envs, n_team, pos_dim)
        comm_range: communication radius; -1 disables distance cutoff
        max_neighbor_slots: padded neighbor dimension for attention

    Returns:
        neighbor_msgs: (n_team, num_envs, max_neighbor_slots, msg_dim)
        neighbor_mask: (n_team, num_envs, max_neighbor_slots)
    """
    n_team, num_envs, msg_dim = all_msgs.shape
    if n_team <= 1:
        return (
            jnp.zeros((n_team, num_envs, max_neighbor_slots, msg_dim)),
            jnp.zeros((n_team, num_envs, max_neighbor_slots)),
        )

    slots = n_team - 1
    agent_idx = jnp.arange(n_team)
    slot_idx = jnp.arange(slots)
    j_for_slot = jnp.where(
        slot_idx[None, :] < agent_idx[:, None], slot_idx, slot_idx + 1
    )

    diff = team_positions[:, :, None, :] - team_positions[:, None, :, :]
    dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-8)

    if comm_range >= 0:
        in_range = dist <= comm_range
    else:
        in_range = jnp.ones_like(dist, dtype=jnp.bool_)
    in_range = in_range & ~jnp.eye(n_team, dtype=jnp.bool_)[None, :, :]

    j_onehot = jax.nn.one_hot(j_for_slot, num_classes=n_team, dtype=all_msgs.dtype)
    neighbor_msgs = jnp.einsum("isj,jed->ised", j_onehot, all_msgs)
    neighbor_msgs = neighbor_msgs.transpose(0, 2, 1, 3)

    j_onehot_f = j_onehot.astype(jnp.float32)
    neighbor_mask = jnp.einsum(
        "isj,eij->eis", j_onehot_f, in_range.astype(jnp.float32)
    ).transpose(1, 0, 2)

    if slots < max_neighbor_slots:
        pad = max_neighbor_slots - slots
        neighbor_msgs = jnp.pad(
            neighbor_msgs, ((0, 0), (0, 0), (0, pad), (0, 0))
        )
        neighbor_mask = jnp.pad(neighbor_mask, ((0, 0), (0, 0), (0, pad)))

    return neighbor_msgs, neighbor_mask


def gather_neighbors_actor_major(
    msgs: FloatArray,
    positions: FloatArray,
    slot_agents: jnp.ndarray,
    env_ids: jnp.ndarray,
    comm_range: float,
    max_neighbor_slots: int,
    n_team: int,
) -> Tuple[FloatArray, FloatArray]:
    """
    Neighbor gather for actor-major batches (e.g. after JaxMARL-style shuffle).

    Only pairs slots from the same parallel env; neighbor slots follow the same
    agent-index ordering as ``gather_neighbors_by_comm_range``.
    """
    batch, msg_dim = msgs.shape
    if n_team <= 1:
        return (
            jnp.zeros((batch, max_neighbor_slots, msg_dim), dtype=msgs.dtype),
            jnp.zeros((batch, max_neighbor_slots), dtype=jnp.float32),
        )

    slots = n_team - 1
    agent_idx = jnp.arange(n_team)
    slot_idx = jnp.arange(slots)
    j_for_slot = jnp.where(
        slot_idx[None, :] < agent_idx[:, None], slot_idx, slot_idx + 1
    )

    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-8)
    if comm_range >= 0:
        in_range = dist <= comm_range
    else:
        in_range = jnp.ones((batch, batch), dtype=jnp.bool_)
    same_env = env_ids[:, None] == env_ids[None, :]

    neighbor_msgs_list = []
    neighbor_mask_list = []
    for s in range(slots):
        target_agent = j_for_slot[slot_agents, s]
        agent_match = slot_agents[None, :] == target_agent[:, None]
        valid_ij = agent_match & same_env & in_range
        weight = valid_ij.astype(msgs.dtype)
        denom = weight.sum(axis=-1, keepdims=True) + 1e-8
        nei_msg = (weight[..., None] * msgs[None, :, :]).sum(axis=1) / denom
        nei_mask = (weight.sum(axis=-1) > 0).astype(jnp.float32)
        neighbor_msgs_list.append(nei_msg)
        neighbor_mask_list.append(nei_mask)

    neighbor_msgs = jnp.stack(neighbor_msgs_list, axis=1)
    neighbor_mask = jnp.stack(neighbor_mask_list, axis=1)
    if slots < max_neighbor_slots:
        pad = max_neighbor_slots - slots
        neighbor_msgs = jnp.pad(
            neighbor_msgs, ((0, 0), (0, pad), (0, 0))
        )
        neighbor_mask = jnp.pad(neighbor_mask, ((0, 0), (0, pad)))
    return neighbor_msgs, neighbor_mask
