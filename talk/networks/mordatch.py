"""Mordatch-style discrete-communication network (MAPPO-GRU backbone).

Faithful adaptation of Mordatch & Abbeel 2017 ("Emergence of Grounded
Compositional Language") to a team-shared MAPPO-GRU actor on JaxMARL MPE.

Channel:
  - Each agent emits a discrete symbol over a fixed vocab via a *soft*
    Gumbel-softmax head (no straight-through; argmax one-hot at deploy/test) -
    matching the reference implementation.
  - Receiver i processes every sender j's previous message with a shared-weight
    GRU keyed by per-(receiver, sender) memory and a sender-id one-hot, then
    max-pools the processed vectors over senders into one comm feature.
  - A grounding auxiliary predicts each sender j's private (sight-masked) obs
    from the per-sender processed vector (one dense layer), replacing the
    paper's goal-prediction loss.

Out-of-range / dead senders contribute a zeroed message to the receive GRU and
are excluded from the obs-prediction loss. Self messages are included in the
pool but excluded from the obs-prediction loss.
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.mlp import _activation_fn


def gumbel_softmax_soft(logits: jnp.ndarray, keys: jax.Array, tau: float) -> jnp.ndarray:
    """Faithful Mordatch Gumbel-softmax: soft simplex sample, no straight-through.

    logits (N, V); keys (N, 2) one PRNG key per row.
    """
    g = jax.vmap(lambda k: jax.random.gumbel(k, (logits.shape[-1],)))(keys)
    return jax.nn.softmax((logits + g) / tau)


class ActorMordatchRNN(nn.Module):
    """GRU actor with Mordatch-style discrete broadcast communication.

    One env step (N agents):
      1. receiver i reads each sender j's prev message through a shared GRU
         (per-(i, j) memory + sender-id), max-pools over senders -> comm_feat
      2. policy GRU consumes concat(obs_embed, comm_feat)
      3. post-update hidden emits action logits and a new (soft) symbol
      4. per-sender processed vectors predict senders' masked obs (grounding)
    """

    action_dim: int
    obs_dim: int
    hidden_size: int = 128
    fc_dim_size: int = 128
    vocab_size: int = 16
    msg_hidden_size: int = 64
    gumbel_tau: float = 1.0
    activation: str = "tanh"

    @nn.compact
    def step(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        prev_msg: jnp.ndarray,
        prev_msg_mem: jnp.ndarray,
        done: jnp.ndarray,
        msg_key: jnp.ndarray,
        comm_reachability: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        dict,
    ]:
        """One communication + policy step for N agents in one env.

        hidden (N, H); obs (N, obs_dim); prev_msg (N, V);
        prev_msg_mem (N_recv, N_send, Hmsg); done (N,); msg_key (N, 2);
        comm_reachability optional (N, N) bool with reach[i, j] = i hears j.

        Returns new_hidden (N, H), new_msg (N, V) soft/hard symbol,
                new_msg_mem (N, N, Hmsg), action_logits (N, action_dim),
                obs_pred_loss (scalar), mean_soft_probs (V,),
                diagnostics (dict of detached scalar metrics).
        """
        activation = _activation_fn(self.activation)
        n = obs.shape[0]
        reset = done.astype(hidden.dtype)
        hidden = jnp.where(reset[:, None], jnp.zeros_like(hidden), hidden)
        alive = ~done.astype(bool)

        # sender_ok[i, j] = receiver i can hear (live) sender j
        sender_ok = alive[None, :]
        if comm_reachability is not None:
            sender_ok = sender_ok & comm_reachability
        sender_ok_f = sender_ok.astype(obs.dtype)

        # receive path: per-(receiver i, sender j) shared GRU over prev messages
        msg_b = prev_msg[None, :, :] * sender_ok_f[:, :, None]  # (N, N, V) zero unheard
        id_b = jnp.broadcast_to(jnp.eye(n, dtype=obs.dtype)[None, :, :], (n, n, n))
        msg_in = jnp.concatenate([msg_b, id_b], axis=-1).reshape(n * n, self.vocab_size + n)

        msg_cell = nn.GRUCell(features=self.msg_hidden_size, name="msg_gru")
        mem_flat = prev_msg_mem.reshape(n * n, self.msg_hidden_size)
        new_mem_flat, p_flat = msg_cell(mem_flat, msg_in)
        new_msg_mem = new_mem_flat.reshape(n, n, self.msg_hidden_size)
        p_ij = p_flat.reshape(n, n, self.msg_hidden_size)  # (N_recv, N_send, Hmsg)

        comm_feat = jnp.max(p_ij, axis=1)  # max-pool over senders -> (N, Hmsg)

        # policy GRU
        obs_embed = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        obs_embed = activation(obs_embed)
        gru_in = jnp.concatenate([obs_embed, comm_feat], axis=-1)

        cell = nn.GRUCell(features=self.hidden_size, name="policy_gru")
        new_hidden, _ = cell(hidden, gru_in)

        # message head (soft Gumbel at train, argmax one-hot at deploy)
        msg_logits = nn.Dense(
            self.vocab_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="msg_head",
        )(new_hidden)
        if deterministic:
            new_msg = jax.nn.one_hot(jnp.argmax(msg_logits, axis=-1), self.vocab_size)
        else:
            new_msg = gumbel_softmax_soft(msg_logits, msg_key, self.gumbel_tau)

        # action head
        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(new_hidden)
        head = activation(head)
        action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)

        # grounding: predict each sender j's masked obs from p_ij (one dense)
        obs_pred = nn.Dense(
            self.obs_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="obs_pred",
        )(p_ij)  # (N_recv, N_send, obs_dim)
        target = jax.lax.stop_gradient(obs[None, :, :])  # sender obs (1, N_send, obs_dim)
        not_self = 1.0 - jnp.eye(n, dtype=obs.dtype)
        pred_mask = sender_ok_f * not_self  # (N_recv, N_send): exclude self/dead/oor
        sq_err = jnp.sum(jnp.square(obs_pred - target), axis=-1)  # (N, N)
        obs_pred_loss = jnp.sum(pred_mask * sq_err) / (jnp.sum(pred_mask) + 1e-8)

        # marginal symbol stats (sparsity penalty computed in trajectory)
        soft_probs = jax.nn.softmax(msg_logits, axis=-1)  # (N, V)
        mean_soft_probs = jnp.mean(soft_probs, axis=0)  # (V,)

        # diagnostics (logging only; detached)
        token_entropy = jnp.mean(
            -jnp.sum(soft_probs * jnp.log(soft_probs + 1e-8), axis=-1)
        )
        comm_norm = jnp.mean(jnp.linalg.norm(comm_feat, axis=-1))
        reachable_frac = jnp.mean(sender_ok_f)
        diagnostics = jax.lax.stop_gradient(
            {
                "token_entropy": token_entropy,
                "comm_norm": comm_norm,
                "reachable_frac": reachable_frac,
                "obs_pred_loss": obs_pred_loss,
            }
        )

        return (
            new_hidden,
            new_msg,
            new_msg_mem,
            action_logits,
            obs_pred_loss,
            mean_soft_probs,
            diagnostics,
        )
