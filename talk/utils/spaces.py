"""Built off Gymnax spaces.py, this module contains jittable classes for action and observation spaces."""

from typing import Tuple
import jax
import jax.numpy as jnp
from talk.utils.typing import PRNGKeyArray, Array

a = jax.Array


class Space(object):
    """
    Minimal jittable class for abstract jaxmarl space.
    """

    def sample(self, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError

    def contains(self, x: Array) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete gymnax spaces.
    """

    def __init__(self, num_categories: int, dtype=jnp.int32):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)

    def contains(self, x: Array) -> bool:  # type: ignore
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """
    Minimal jittable class for array-shaped gymnax spaces.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(
            self.dtype
        )

    def contains(self, x: Array) -> bool:  # type: ignore
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond
