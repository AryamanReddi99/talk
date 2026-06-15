"""TarMAC communication networks (environment-agnostic)."""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.mlp import _activation_fn


def tarmac_aggregate(
    query: jnp.ndarray,
    signature: jnp.ndarray,
    value: jnp.ndarray,
    alive_mask: jnp.ndarray,
    sender_reachable: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    TarMAC soft attention over all senders in one env.

    query: (N, sig_dim), signature: (N, sig_dim), value: (N, val_dim)
    alive_mask: (N,) — False masks out dead senders
    sender_reachable: optional (N, N) bool — True if receiver j can hear sender i
    returns comm_ctx: (N, val_dim) — one aggregated message per receiver
    """
    sig_dim = query.shape[-1]
    scale = jnp.sqrt(jnp.asarray(sig_dim, dtype=query.dtype))
    scores = jnp.einsum("jd,id->ji", query, signature) / scale
    large_neg = jnp.finfo(scores.dtype).min
    sender_ok = alive_mask[None, :]
    if sender_reachable is not None:
        sender_ok = sender_ok & sender_reachable
    scores = jnp.where(sender_ok, scores, large_neg)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("ji,id->jd", weights, value)


class ActorTarMACRNN(nn.Module):
    """
    GRU actor with TarMAC communication (signature / value / query attention).

    One env step (N agents):
      1. query from pre-update hidden attends to prev (signature, value)
      2. GRU consumes concat(obs, comm_ctx)
      3. post-update hidden emits new signature, value, and action logits
    """

    action_dim: int
    hidden_size: int = 64
    fc_dim_size: int = 64
    sig_dim: int = 16
    val_dim: int = 32
    activation: str = "tanh"

    @nn.compact
    def step(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        prev_signature: jnp.ndarray,
        prev_value: jnp.ndarray,
        done: jnp.ndarray,
        comm_reachability: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        hidden: (N, H), obs: (N, obs_dim)
        prev_signature: (N, sig_dim), prev_value: (N, val_dim)
        done: (N,)
        comm_reachability: optional (N, N) bool — range-limited sender mask
        returns new_hidden (N, H), signature (N, sig_dim), value (N, val_dim),
                logits (N, action_dim)
        """
        activation = _activation_fn(self.activation)
        reset = done.astype(hidden.dtype)
        hidden = jnp.where(reset[:, None], jnp.zeros_like(hidden), hidden)
        alive = ~done.astype(bool)

        query = nn.Dense(
            self.sig_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="query",
        )(hidden)
        comm_ctx = tarmac_aggregate(
            query, prev_signature, prev_value, alive, sender_reachable=comm_reachability
        )

        obs_embed = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        obs_embed = activation(obs_embed)
        gru_in = jnp.concatenate([obs_embed, comm_ctx], axis=-1)

        cell = nn.GRUCell(features=self.hidden_size)
        new_hidden, _ = cell(hidden, gru_in)

        signature = nn.Dense(
            self.sig_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="signature",
        )(new_hidden)
        value = nn.Dense(
            self.val_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="value",
        )(new_hidden)

        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(new_hidden)
        head = activation(head)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)
        return new_hidden, signature, value, logits
