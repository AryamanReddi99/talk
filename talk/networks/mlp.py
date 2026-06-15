import flax.linen as nn
from typing import Sequence, Tuple
from flax.linen.initializers import constant, orthogonal
import numpy as np
import jax.numpy as jnp
from distrax import Categorical, Joint, MultivariateNormalDiag


def _activation_fn(name: str):
    if name == "relu":
        return nn.relu
    if name == "elu":
        return nn.elu
    if name == "tanh":
        return nn.tanh
    raise ValueError(f"Unknown activation function: {name}")


class ActorContinuous(nn.Module):
    """Diagonal Gaussian policy for continuous control (mean from MLP, learnable log_std)."""

    action_dim: int
    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = _activation_fn(self.activation)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        h = activation(h)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(h)
        h = activation(h)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        # State-independent log_std (common in PPO; stable-baselines3-style).
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))


class CriticContinuous(nn.Module):
    """Value network for continuous control (same MLP head as discrete critic)."""

    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = _activation_fn(self.activation)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorCriticContinuous(nn.Module):
    """Legacy combined module: Gaussian actor + value (same shapes as separate Actor+Critic)."""

    action_dim: int
    activation: str = "tanh"
    hidden_dim: int = 64

    def setup(self):
        self.actor = ActorContinuous(
            action_dim=self.action_dim,
            activation=self.activation,
            hidden_dim=self.hidden_dim,
        )
        self.critic = CriticContinuous(
            activation=self.activation,
            hidden_dim=self.hidden_dim,
        )

    def __call__(self, x):
        return self.actor(x), self.critic(x)


class ActorDiscrete(nn.Module):
    """Policy network for discrete actions."""

    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = _activation_fn(self.activation)
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        return Categorical(logits=actor_mean)


class ActorMultiDiscrete(nn.Module):
    """Independent categoricals over each dimension (Jumanji ``MultiDiscreteArray``)."""

    num_values: Tuple[int, ...]
    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = _activation_fn(self.activation)
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        flat = nn.Dense(
            sum(self.num_values),
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        cats = []
        pos = 0
        for nv in self.num_values:
            cats.append(Categorical(logits=flat[:, pos : pos + nv]))
            pos += nv
        return Joint(tuple(cats))


class CriticDiscrete(nn.Module):
    """Critic network"""

    activation: str = "tanh"
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = _activation_fn(self.activation)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorCriticDiscrete(nn.Module):
    """Shared wrapper: actor + critic (for scripts that still use one module)."""

    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 64

    def setup(self):
        self.actor = ActorDiscrete(
            action_dim=self.action_dim,
            activation=self.activation,
            hidden_dim=self.hidden_dim,
        )
        self.critic = CriticDiscrete(
            activation=self.activation,
            hidden_dim=self.hidden_dim,
        )

    def __call__(self, x):
        return self.actor(x), self.critic(x)
