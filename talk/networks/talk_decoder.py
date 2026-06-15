"""Autoregressive discrete-token communication network (Talk-Gumbel).

Replaces the continuous TarMAC (signature, value) channel with an LLM-style
decoder that emits a variable-length token message via the Gumbel
straight-through estimator. See
``talk/experiments/mpe/talk_gumbel/ARCHITECTURE.md`` for the full design.
"""

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.mlp import _activation_fn

BOS_IDX = 0
EOS_IDX = 1


def sinusoidal_pos_enc(length: int, dim: int) -> jnp.ndarray:
    """Fixed sinusoidal positional table of shape (length, dim)."""
    pos = jnp.arange(length)[:, None]
    idx = jnp.arange(dim)[None, :]
    angle_rates = 1.0 / jnp.power(10000.0, (2 * (idx // 2)) / jnp.asarray(dim, jnp.float32))
    angles = pos * angle_rates
    return jnp.where(idx % 2 == 0, jnp.sin(angles), jnp.cos(angles))


def st_gumbel(logits: jnp.ndarray, keys: jax.Array, tau: float) -> jnp.ndarray:
    """Gumbel straight-through: forward hard one-hot, backward soft softmax.

    logits (N, V); keys (N, 2) one PRNG key per row.
    """
    g = jax.vmap(lambda k: jax.random.gumbel(k, (logits.shape[-1],)))(keys)
    y_soft = jax.nn.softmax((logits + g) / tau)
    y_hard = jax.nn.one_hot(jnp.argmax(logits + g, axis=-1), logits.shape[-1])
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


class ActorTalkRNN(nn.Module):
    """GRU actor with autoregressive discrete-token communication.

    One env step (N agents):
      1. query_listen from pre-update hidden attends over all agents' prev tokens
      2. GRU consumes concat(obs_embed, comm_ctx)
      3. post-update hidden emits action logits and an autoregressive message
    """

    action_dim: int
    hidden_size: int = 128
    fc_dim_size: int = 128
    vocab_content: int = 10
    vocab_embed_dim: int = 64
    attn_dim: int = 32
    max_msg_len: int = 10
    min_msg_len: int = 0
    gumbel_tau: float = 1.0
    activation: str = "tanh"

    @property
    def vocab_size(self) -> int:
        return self.vocab_content + 2

    def setup(self):
        v = self.vocab_size
        d_tok = self.vocab_embed_dim
        d_att = self.attn_dim

        self.token_embed = self.param("token_embed", orthogonal(1.0), (v, d_tok))
        self.pos_table = sinusoidal_pos_enc(self.max_msg_len + 1, d_tok)

        dense = lambda feat, scale, name: nn.Dense(
            feat,
            kernel_init=orthogonal(scale),
            bias_init=constant(0.0),
            name=name,
        )

        # shared token key/value projections (decode + listen)
        self.k_tok = dense(d_att, np.sqrt(2), "k_tok")
        self.v_tok = dense(d_att, np.sqrt(2), "v_tok")

        # decoder-only projections
        self.q_dec = dense(d_att, np.sqrt(2), "q_dec")
        self.k_ctx = dense(d_att, np.sqrt(2), "k_ctx")
        self.v_ctx = dense(d_att, np.sqrt(2), "v_ctx")
        self.o_dec = dense(d_tok, np.sqrt(2), "o_dec")
        self.dec_mlp_h = dense(self.fc_dim_size, np.sqrt(2), "dec_mlp_h")
        self.dec_mlp_out = dense(v, 0.01, "dec_mlp_out")

        # listener-only query projection
        self.q_listen = dense(d_att, np.sqrt(2), "q_listen")

        # policy backbone
        self.obs_embed = dense(self.fc_dim_size, np.sqrt(2), "obs_embed")
        self.cell = nn.GRUCell(features=self.hidden_size, name="gru")
        self.act_h = dense(self.fc_dim_size, 2, "act_h")
        self.act_out = dense(self.action_dim, 0.01, "act_out")

    def _embed_tokens(self, onehots: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        """onehots (..., V) -> (..., d_tok) with sinusoidal positions added.

        positions broadcasts against the leading dims of onehots.
        """
        emb = onehots @ self.token_embed
        return emb + self.pos_table[positions]

    def _listen(
        self,
        hidden: jnp.ndarray,
        prev_tokens: jnp.ndarray,
        prev_valid: jnp.ndarray,
        alive: jnp.ndarray,
        reachability: jnp.ndarray,
    ) -> jnp.ndarray:
        """Flat masked attention over all agents' previous tokens (incl. self).

        hidden (N, H); prev_tokens (N, L, V); prev_valid (N, L); alive (N,);
        reachability (N, N). Returns comm_ctx (N, attn_dim).
        """
        n = hidden.shape[0]
        length = self.max_msg_len
        slot_pos = jnp.arange(1, length + 1)

        query = self.q_listen(hidden)  # (N, d_att)
        embeds = self._embed_tokens(prev_tokens, slot_pos[None, :])  # (N, L, d_tok)
        keys = self.k_tok(embeds).reshape(n * length, self.attn_dim)
        values = self.v_tok(embeds).reshape(n * length, self.attn_dim)

        scale = jnp.sqrt(jnp.asarray(self.attn_dim, dtype=query.dtype))
        scores = jnp.einsum("jd,kd->jk", query, keys) / scale  # (N, N*L)

        mask = (
            reachability[:, :, None]
            & alive[None, :, None]
            & prev_valid.astype(bool)[None, :, :]
        ).reshape(n, n * length)
        large_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask, scores, large_neg)
        weights = jax.nn.softmax(scores, axis=-1)
        comm_ctx = jnp.einsum("jk,kd->jd", weights, values)
        any_valid = jnp.any(mask, axis=-1, keepdims=True)
        return jnp.where(any_valid, comm_ctx, 0.0)

    def _decode(
        self,
        hidden: jnp.ndarray,
        token_keys: jnp.ndarray,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Autoregressive message decode conditioned on hidden (post-GRU).

        hidden (N, H); token_keys (N, L, 2). Returns msg_tokens (N, L, V),
        msg_valid (N, L), expected_len (N,).
        """
        n = hidden.shape[0]
        v = self.vocab_size
        large_neg = jnp.finfo(jnp.float32).min

        k_ctx = self.k_ctx(hidden)[:, None, :]  # (N, 1, d_att)
        v_ctx = self.v_ctx(hidden)[:, None, :]

        bos = jax.nn.one_hot(jnp.full((n,), BOS_IDX), v)
        eos = jax.nn.one_hot(jnp.full((n,), EOS_IDX), v)

        ctx_embeds = [self._embed_tokens(bos, jnp.zeros((n,), jnp.int32))]
        done = jnp.zeros((n,), dtype=bool)
        survival = jnp.ones((n,))
        expected_len = jnp.zeros((n,))
        tokens_out = []
        valid_out = []

        for t in range(self.max_msg_len):
            query = self.q_dec(ctx_embeds[-1])  # (N, d_att)
            ctx_stack = jnp.stack(ctx_embeds, axis=1)  # (N, t+1, d_tok)
            keys = jnp.concatenate([k_ctx, self.k_tok(ctx_stack)], axis=1)
            values = jnp.concatenate([v_ctx, self.v_tok(ctx_stack)], axis=1)
            scale = jnp.sqrt(jnp.asarray(self.attn_dim, dtype=query.dtype))
            scores = jnp.einsum("nd,nkd->nk", query, keys) / scale
            weights = jax.nn.softmax(scores, axis=-1)
            attn = jnp.einsum("nk,nkd->nd", weights, values)
            out = self.o_dec(attn)
            head = _activation_fn(self.activation)(self.dec_mlp_h(out))
            logits = self.dec_mlp_out(head)
            logits = logits.at[:, BOS_IDX].set(large_neg)  # never emit BOS
            if t < self.min_msg_len:
                logits = logits.at[:, EOS_IDX].set(large_neg)  # force >=min_msg_len

            p_eos = jax.nn.softmax(logits, axis=-1)[:, EOS_IDX]
            expected_len = expected_len + survival
            survival = survival * (1.0 - p_eos)

            if deterministic:
                token = jax.nn.one_hot(jnp.argmax(logits, axis=-1), v)
            else:
                token = st_gumbel(logits, token_keys[:, t], self.gumbel_tau)
            is_eos = jnp.argmax(token, axis=-1) == EOS_IDX

            valid = (~done) & (~is_eos)
            tokens_out.append(jnp.where(valid[:, None], token, 0.0))
            valid_out.append(valid)

            ar_tok = jnp.where(done[:, None], eos, token)
            next_emb = self._embed_tokens(
                ar_tok, jnp.full((n,), t + 1, jnp.int32)
            )
            new_done = done | is_eos
            next_emb = jnp.where(
                new_done[:, None], jax.lax.stop_gradient(next_emb), next_emb
            )
            ctx_embeds.append(next_emb)
            done = new_done

        msg_tokens = jnp.stack(tokens_out, axis=1)  # (N, L, V)
        msg_valid = jnp.stack(valid_out, axis=1)  # (N, L)
        return msg_tokens, msg_valid, expected_len

    def step(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        prev_tokens: jnp.ndarray,
        prev_valid: jnp.ndarray,
        done: jnp.ndarray,
        reachability: jnp.ndarray,
        msg_key: jnp.ndarray,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One communication + policy step for N agents in one env.

        hidden (N, H); obs (N, obs_dim); prev_tokens (N, L, V);
        prev_valid (N, L); done (N,); reachability (N, N); msg_key (N, 2).
        Returns new_hidden (N, H), action_logits (N, action_dim),
                msg_tokens (N, L, V), msg_valid (N, L), expected_len (N,),
                comm_ctx (N, attn_dim).
        """
        activation = _activation_fn(self.activation)
        reset = done.astype(hidden.dtype)
        hidden = jnp.where(reset[:, None], jnp.zeros_like(hidden), hidden)
        alive = ~done.astype(bool)

        comm_ctx = self._listen(hidden, prev_tokens, prev_valid, alive, reachability)

        obs_embed = activation(self.obs_embed(obs))
        gru_in = jnp.concatenate([obs_embed, comm_ctx], axis=-1)
        new_hidden, _ = self.cell(hidden, gru_in)

        head = activation(self.act_h(new_hidden))
        logits = self.act_out(head)

        token_keys = jax.vmap(lambda k: jax.random.split(k, self.max_msg_len))(msg_key)
        msg_tokens, msg_valid, expected_len = self._decode(
            new_hidden, token_keys, deterministic
        )
        return new_hidden, logits, msg_tokens, msg_valid, expected_len, comm_ctx
