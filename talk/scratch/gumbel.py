"""Minimal demo: backprop through autoregressive ST-Gumbel token generation.

Task: each context vector c_i must be identified from L discrete tokens the
speaker autoregressively emits while attending to a constant context embedding.
Loss is batch InfoNCE applied only after the final token — no per-step supervision.

Run:
    python talk/scratch/gumbel.py --seq-len 6
"""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import optax


def st_gumbel(logits: jnp.ndarray, key: jax.Array, tau: float = 1.0) -> jnp.ndarray:
    g = jax.random.gumbel(key, logits.shape)
    y_soft = jax.nn.softmax((logits + g) / tau)
    y_hard = jax.nn.one_hot(jnp.argmax(logits + g, axis=-1), logits.shape[-1])
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


def init_params(key: jax.Array, vocab: int, hidden: int) -> dict[str, jnp.ndarray]:
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    scale = lambda k, shape: jax.random.normal(k, shape) * 0.1
    return {
        "ctx_proj": scale(k1, (hidden, hidden)),
        "tok_embed": scale(k2, (vocab, hidden)),
        "step_w": scale(k3, (hidden, hidden)),
        "ctx_w": scale(k4, (hidden, hidden)),
        "step_b": jnp.zeros((hidden,)),
        "out_w": scale(k5, (hidden, vocab)),
        "out_b": jnp.zeros((vocab,)),
        "listen_w": scale(k6, (hidden, hidden)),
        "listen_b": jnp.zeros((hidden,)),
    }


def encode(params: dict[str, jnp.ndarray], ctx: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(ctx @ params["ctx_proj"])


def speaker_step(
    params: dict[str, jnp.ndarray],
    prev_onehot: jnp.ndarray,
    ctx_emb: jnp.ndarray,
    key: jax.Array,
    tau: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = prev_onehot @ params["tok_embed"]
    h = jnp.tanh(x @ params["step_w"] + ctx_emb @ params["ctx_w"] + params["step_b"])
    logits = h @ params["out_w"] + params["out_b"]
    return st_gumbel(logits, key, tau), logits


def speak(
    params: dict[str, jnp.ndarray],
    ctx_emb: jnp.ndarray,
    key: jax.Array,
    seq_len: int,
    tau: float,
    stop_after: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch, vocab = ctx_emb.shape[0], params["tok_embed"].shape[0]
    bos = jnp.zeros((batch, vocab)).at[:, 0].set(1.0)

    def step(
        carry: jnp.ndarray,
        xs: tuple[jnp.ndarray, jax.Array],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        step_idx, key_t = xs
        if stop_after is not None:
            carry = jax.lax.cond(
                step_idx > stop_after,
                lambda x: jax.lax.stop_gradient(x),
                lambda x: x,
                carry,
            )
        tok, logits = speaker_step(params, carry, ctx_emb, key_t, tau)
        return tok, (tok, logits)

    keys = jax.random.split(key, seq_len)
    _, (tokens, logits) = jax.lax.scan(
        step,
        bos,
        (jnp.arange(seq_len), keys),
    )
    return tokens, logits


def listen(params: dict[str, jnp.ndarray], tokens: jnp.ndarray) -> jnp.ndarray:
    # tokens: (L, B, V) hard one-hots in the forward pass
    embeds = tokens @ params["tok_embed"]
    msg = jnp.mean(embeds, axis=0)
    return jnp.tanh(msg @ params["listen_w"] + params["listen_b"])


def contrastive_loss(msg: jnp.ndarray, ctx_emb: jnp.ndarray) -> jnp.ndarray:
    scores = msg @ ctx_emb.T / jnp.sqrt(msg.shape[-1])
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    return -jnp.mean(jnp.diag(log_probs))


def loss_fn(
    params: dict[str, jnp.ndarray],
    ctx: jnp.ndarray,
    key: jax.Array,
    seq_len: int,
    tau: float,
    stop_after: int | None = None,
    listen_last: bool = False,
) -> jnp.ndarray:
    ctx_emb = encode(params, ctx)
    tokens, _ = speak(params, ctx_emb, key, seq_len, tau, stop_after=stop_after)
    msg = listen(params, tokens[-1:] if listen_last else tokens)
    return contrastive_loss(msg, ctx_emb)


def batch_accuracy(
    params: dict[str, jnp.ndarray],
    ctx: jnp.ndarray,
    key: jax.Array,
    seq_len: int,
) -> jnp.ndarray:
    ctx_emb = encode(params, ctx)
    tokens, _ = speak(params, ctx_emb, key, seq_len, tau=0.1)
    msg = listen(params, tokens)
    preds = jnp.argmax(msg @ ctx_emb.T, axis=-1)
    return jnp.mean(preds == jnp.arange(ctx.shape[0]))


def grad_norm(params: dict[str, jnp.ndarray]) -> float:
    leaves = jax.tree_util.tree_leaves(params)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in leaves)))


def grad_norm_for_leaf(params: dict[str, jnp.ndarray], name: str) -> float:
    return float(jnp.linalg.norm(params[name]))


def train(args: argparse.Namespace) -> None:
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, args.vocab, args.hidden)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(
        params: dict[str, jnp.ndarray],
        opt_state: Any,
        ctx: jnp.ndarray,
        step_key: jax.Array,
    ) -> tuple[dict[str, jnp.ndarray], Any, jnp.ndarray, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, ctx, step_key, args.seq_len, args.tau)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        acc = batch_accuracy(params, ctx, step_key, args.seq_len)
        return params, opt_state, loss, acc

    chance = 1.0 / args.batch
    print(f"seq_len={args.seq_len}  vocab={args.vocab}  batch={args.batch}  chance={chance:.4f}")
    print("step    loss      acc     ||grad||")
    print("-" * 40)

    for step in range(1, args.steps + 1):
        key, ctx_key, step_key = jax.random.split(key, 3)
        ctx = jax.random.normal(ctx_key, (args.batch, args.hidden))
        params, opt_state, loss, acc = train_step(params, opt_state, ctx, step_key)
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            gn = grad_norm(jax.grad(loss_fn)(params, ctx, step_key, args.seq_len, args.tau))
            print(f"{step:4d}  {float(loss):8.4f}  {float(acc):6.3f}  {gn:8.4f}")

    key, ctx_key, step_key, probe_key = jax.random.split(key, 4)
    ctx = jax.random.normal(ctx_key, (args.batch, args.hidden))

    print("\nChain ablation (listen-last-only; block carry gradient after step k):")
    print("  stop_after   ||grad||   ||grad step_w||")
    for stop in [None, *range(args.seq_len - 1)]:
        grads = jax.grad(loss_fn)(
            params,
            ctx,
            probe_key,
            args.seq_len,
            args.tau,
            stop_after=stop,
            listen_last=False,
        )
        label = "none" if stop is None else str(stop)
        print(
            f"  {label:>11}  {grad_norm(grads):8.4f}  "
            f"{grad_norm_for_leaf(grads, 'step_w'):8.4f}"
        )
    print(
        "Only the final token is read out, so earlier steps reach the loss through "
        "the autoregressive carry. Blocking the carry after step k removes steps "
        "0..k from that chain."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=6, help="autoregressive length L")
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--tau", type=float, default=1.0, help="Gumbel-Softmax temperature")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
