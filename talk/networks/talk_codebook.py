"""Per-agent codebook communication network (Talk Codebook v2).

TarMAC-style attention where the transmitted value is a Gumbel-selected row of a
per-agent, hidden-conditioned codebook. Keys/signatures and the discrete index
are available to receivers, so attention routing matches TarMAC; only the
attended *value* changes.

Two comm modes:
  - sender (training): receiver j aggregates sum_i a_ji * (k_i @ e_i)  (sender row)
  - receiver (test/deploy): receiver j aggregates sum_i a_ji * (k_i @ e_j)
    using its own codebook at the received index.

An auxiliary, attention-weighted L2 loss pulls each receiver's embedding e_j at a
received index toward the sender's used row (stop-gradient on the sender side, and
on the attention weight), so the receiver lookup converges toward the sender one.
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.mlp import _activation_fn
from talk.networks.talk_decoder import st_gumbel


class ActorTalkCodebookRNN(nn.Module):
    """GRU actor with codebook communication (signature / query attention +
    Gumbel-selected codebook-row value).

    One env step (N agents):
      1. query from pre-update hidden attends to prev (signature) -> weights a_ji
      2. value = Gumbel-selected row of sender (or receiver) codebook; comm = a @ value
      3. GRU consumes concat(obs, comm)
      4. post-update hidden emits new signature, codebook, selection logits, action logits
    """

    action_dim: int
    hidden_size: int = 128
    fc_dim_size: int = 128
    sig_dim: int = 16
    codebook_size: int = 16
    vocab_dim: int = 32
    gumbel_tau: float = 1.0
    activation: str = "tanh"

    @nn.compact
    def step(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        prev_signature: jnp.ndarray,
        prev_codebook: jnp.ndarray,
        prev_onehot: jnp.ndarray,
        done: jnp.ndarray,
        msg_key: jnp.ndarray,
        comm_reachability: Optional[jnp.ndarray] = None,
        receiver_lookup: bool = False,
        deterministic: bool = False,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        dict,
    ]:
        """One communication + policy step for N agents in one env.

        hidden (N, H); obs (N, obs_dim); prev_signature (N, sig_dim);
        prev_codebook (N, M, d_vocab); prev_onehot (N, M); done (N,);
        msg_key (N, 2); comm_reachability optional (N, N) bool.

        Returns new_hidden (N, H), signature (N, sig_dim),
                codebook (N, M, d_vocab), onehot (N, M), action_logits (N, action_dim),
                aux_loss (scalar), comm_ctx (N, d_vocab),
                diagnostics (dict of detached scalar metrics for logging).
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

        scale = jnp.sqrt(jnp.asarray(self.sig_dim, dtype=query.dtype))
        scores = jnp.einsum("jd,id->ji", query, prev_signature) / scale
        sender_ok = alive[None, :]
        if comm_reachability is not None:
            sender_ok = sender_ok & comm_reachability
        large_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(sender_ok, scores, large_neg)
        attn = jax.nn.softmax(scores, axis=-1)  # (N, N): attn[j, i]

        # sender row for each sender i: k_i @ e_i  (gradient flows to sender)
        sender_rows = jnp.einsum("im,imd->id", prev_onehot, prev_codebook)  # (N, d_vocab)
        # receiver row for each (receiver j, sender i): stop_grad(k_i) @ e_j
        onehot_sg = jax.lax.stop_gradient(prev_onehot)
        recv_rows = jnp.einsum("im,jmd->jid", onehot_sg, prev_codebook)  # (N, N, d_vocab)

        if receiver_lookup:
            comm_ctx = jnp.einsum("ji,jid->jd", attn, recv_rows)
        else:
            comm_ctx = jnp.einsum("ji,id->jd", attn, sender_rows)

        # auxiliary alignment loss: move receiver embedding e_j[k_i] toward sender's
        # used row, weighted by (detached) attention. Uses prev-step codebooks.
        target = jax.lax.stop_gradient(sender_rows)  # (N, d_vocab) over sender i
        diff = recv_rows - target[None, :, :]  # (j, i, d)
        dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-8)  # (j, i)
        weight = jax.lax.stop_gradient(attn) * sender_ok.astype(attn.dtype)  # (j, i)
        aux_per_receiver = jnp.sum(weight * dist, axis=-1)  # (j,)
        aux_loss = jnp.mean(aux_per_receiver)

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
        codebook = nn.Dense(
            self.codebook_size * self.vocab_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="codebook",
        )(new_hidden)
        codebook = codebook.reshape(-1, self.codebook_size, self.vocab_dim)

        sel_logits = nn.Dense(
            self.codebook_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="select",
        )(new_hidden)

        if deterministic:
            onehot = jax.nn.one_hot(jnp.argmax(sel_logits, axis=-1), self.codebook_size)
        else:
            onehot = st_gumbel(sel_logits, msg_key, self.gumbel_tau)

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

        # diagnostics (logging only; detached so they never affect gradients)
        valid = sender_ok.astype(attn.dtype)
        raw_align_l2 = jnp.sum(dist * valid) / (jnp.sum(valid) + 1e-8)
        sel_probs = jax.nn.softmax(sel_logits, axis=-1)
        token_entropy = jnp.mean(
            -jnp.sum(sel_probs * jnp.log(sel_probs + 1e-8), axis=-1)
        )
        attention_entropy = jnp.mean(
            -jnp.sum(attn * jnp.log(attn + 1e-8), axis=-1)
        )
        self_attention = jnp.mean(jnp.diagonal(attn))
        comm_norm = jnp.mean(jnp.linalg.norm(comm_ctx, axis=-1))

        # codebook row diversity: mean off-diagonal cosine sim within each codebook
        cb_norm = codebook / (
            jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8
        )
        gram = jnp.einsum("nmd,nld->nml", cb_norm, cb_norm)
        off_diag = gram - jnp.eye(self.codebook_size)[None, :, :]
        codebook_intra_cosine = jnp.sum(off_diag) / (
            codebook.shape[0] * self.codebook_size * (self.codebook_size - 1) + 1e-8
        )

        # hidden -> codebook sensitivity: cross-agent spread ratio within one env
        h_mean = jnp.mean(new_hidden, axis=0, keepdims=True)
        cb_mean = jnp.mean(codebook, axis=0, keepdims=True)
        h_spread = jnp.mean(jnp.linalg.norm(new_hidden - h_mean, axis=-1))
        cb_spread = jnp.mean(
            jnp.linalg.norm(
                codebook.reshape(codebook.shape[0], -1)
                - cb_mean.reshape(1, -1),
                axis=-1,
            )
        )
        codebook_hidden_sensitivity = cb_spread / (h_spread + 1e-8)

        diagnostics = jax.lax.stop_gradient(
            {
                "raw_align_l2": raw_align_l2,
                "token_entropy": token_entropy,
                "attention_entropy": attention_entropy,
                "self_attention": self_attention,
                "comm_norm": comm_norm,
                "codebook_intra_cosine": codebook_intra_cosine,
                "codebook_hidden_sensitivity": codebook_hidden_sensitivity,
            }
        )

        return (
            new_hidden,
            signature,
            codebook,
            onehot,
            action_logits,
            aux_loss,
            comm_ctx,
            diagnostics,
        )
