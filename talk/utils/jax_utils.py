import jax
import jax.numpy as jnp
from talk.utils.typing import PyTree, Array


def pytree_norm(pytree: PyTree) -> Array:
    """
    Computes the L2 norm of a pytree
    """
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), pytree)
    total_square = jax.tree.reduce(lambda leaf_1, leaf_2: leaf_1 + leaf_2, squares)
    return jnp.sqrt(total_square)


jprint = lambda *args: [jax.debug.print("{var}", var=arg) for arg in args]


def flatten_pytree(pytree: PyTree) -> Array:
    """Concatenate all leaves into one vector."""
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(pytree)])


def cosine_similarity_vectors(vec1: Array, vec2: Array) -> Array:
    """Cosine similarity between two 1D vectors."""
    dot = jnp.dot(vec1, vec2)
    denom = jnp.linalg.norm(vec1) * jnp.linalg.norm(vec2)
    return jnp.where(denom > 1e-8, dot / denom, 0.0)


def batch_cosine_similarity_stats(vectors: Array) -> dict[str, Array]:
    """Mean / min / max pairwise cosine similarity for rows of ``vectors`` (K, D)."""

    k = vectors.shape[0]
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    g = vectors / (norms + 1e-8)
    gram = g @ g.T
    mask = 1.0 - jnp.eye(k, dtype=gram.dtype)
    denom = jnp.maximum(mask.sum(), 1.0)
    off_diag_sum = (gram * mask).sum()
    mean_cos = jnp.where(k > 1, off_diag_sum / denom, 0.0)
    min_cos = jnp.where(
        k > 1,
        jnp.min(jnp.where(mask > 0, gram, 2.0)),
        0.0,
    )
    max_cos = jnp.where(
        k > 1,
        jnp.max(jnp.where(mask > 0, gram, -2.0)),
        0.0,
    )
    return {"mean": mean_cos, "min": min_cos, "max": max_cos}
