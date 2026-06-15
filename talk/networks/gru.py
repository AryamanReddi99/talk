"""GRU actor/critic networks for MAPPO (JaxMARL-style ScannedRNN)."""

import functools
from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.mlp import _activation_fn


class ScannedRNN(nn.Module):
    """Time-scanned GRU cell with done-masked state reset (JaxMARL MAPPO)."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state, (inputs, resets) = carry, x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, output = nn.GRUCell(features=inputs.shape[-1])(rnn_state, inputs)
        return new_rnn_state, output

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorDiscreteRNN(nn.Module):
    """Discrete policy: Dense -> activation -> GRU -> Dense -> activation -> logits."""

    action_dim: int
    hidden_size: int = 64
    fc_dim_size: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, distrax.Distribution]:
        obs, dones = x
        activation = _activation_fn(self.activation)
        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        head = activation(head)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)
        return hidden, distrax.Categorical(logits=logits)


class CriticDiscreteRNN(nn.Module):
    """State-conditioned value: Dense -> activation -> GRU -> Dense -> activation -> scalar."""

    hidden_size: int = 64
    fc_dim_size: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        state, dones = x
        activation = _activation_fn(self.activation)
        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(state)
        embedding = activation(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        head = activation(head)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(head)
        return hidden, jnp.squeeze(value, axis=-1)


class _ActorGRUEncoder(nn.Module):
    """Dense -> GRU; outputs carry and message (GRU output)."""

    fc_dim_size: int
    activation: str

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        activation = _activation_fn(self.activation)
        embedding = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)
        return ScannedRNN()(hidden, (embedding, dones))


class _ActorCommHead(nn.Module):
    """Policy head on concat(own message, neighbor messages)."""

    action_dim: int
    fc_dim_size: int
    neighbor_msg_dim: int
    activation: str

    @nn.compact
    def __call__(
        self,
        message: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
    ) -> distrax.Distribution:
        activation = _activation_fn(self.activation)
        if self.neighbor_msg_dim > 0:
            head_in = jnp.concatenate([message, neighbor_msgs], axis=-1)
        else:
            head_in = message
        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(head_in)
        head = activation(head)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)
        return distrax.Categorical(logits=logits)


class _MaskedNeighborAttention(nn.Module):
    """Single-head attention: query from own message, keys/values from neighbors."""

    msg_dim: int

    @nn.compact
    def __call__(
        self,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        query = nn.Dense(
            self.msg_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="q",
        )(own_msg)
        keys = nn.Dense(
            self.msg_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="k",
        )(neighbor_msgs)
        values = nn.Dense(
            self.msg_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="v",
        )(neighbor_msgs)

        scale = jnp.sqrt(jnp.asarray(self.msg_dim, dtype=own_msg.dtype))
        scores = jnp.einsum("...d,...nd->...n", query, keys) / scale
        large_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(neighbor_mask > 0, scores, large_neg)
        weights = jax.nn.softmax(scores, axis=-1)
        weights = weights * neighbor_mask
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        return jnp.einsum("...n,...nd->...d", weights, values)


class _ActorCommAttnHead(nn.Module):
    """Policy head: attention over neighbors, then concat(own_msg, context)."""

    action_dim: int
    fc_dim_size: int
    msg_dim: int
    activation: str

    @nn.compact
    def __call__(
        self,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> distrax.Distribution:
        activation = _activation_fn(self.activation)
        context = _MaskedNeighborAttention(self.msg_dim)(
            own_msg, neighbor_msgs, neighbor_mask
        )
        head_in = jnp.concatenate([own_msg, context], axis=-1)
        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(head_in)
        head = activation(head)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)
        return distrax.Categorical(logits=logits)


class ActorDiscreteCommAttnRNN(nn.Module):
    """
    Communicating discrete policy with masked neighbor attention.

    Encoder produces a message; head attends over neighbor messages (fixed slots
    with mask) and consumes concat(own message, attention context).

    Use ``encode`` / ``act`` via ``apply(..., method=...)`` in the training loop.
    """

    action_dim: int
    fc_dim_size: int = 64
    msg_dim: int = 64
    activation: str = "tanh"

    def setup(self):
        self.encoder = _ActorGRUEncoder(
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )
        self.head = _ActorCommAttnHead(
            action_dim=self.action_dim,
            fc_dim_size=self.fc_dim_size,
            msg_dim=self.msg_dim,
            activation=self.activation,
        )

    def encode(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder(hidden, obs, dones)

    def act(
        self,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
    ) -> distrax.Distribution:
        return self.head(own_msg, neighbor_msgs, neighbor_mask)

    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution]:
        obs, dones, neighbor_msgs, neighbor_mask = x
        hidden, message = self.encode(hidden, obs, dones)
        pi = self.act(message, neighbor_msgs, neighbor_mask)
        return hidden, message, pi


class _ObsHiddenMessageLayer(nn.Module):
    """concat(obs, hidden) -> Dense -> activation -> message."""

    msg_dim: int
    activation: str

    @nn.compact
    def __call__(self, obs: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        activation = _activation_fn(self.activation)
        x = jnp.concatenate([obs, hidden], axis=-1)
        msg = nn.Dense(
            self.msg_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return activation(msg)


class _ActorGRUActionHead(nn.Module):
    """Post-GRU policy head: Dense -> activation -> logits."""

    action_dim: int
    fc_dim_size: int
    activation: str

    @nn.compact
    def __call__(self, gru_out: jnp.ndarray) -> distrax.Distribution:
        activation = _activation_fn(self.activation)
        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(gru_out)
        head = activation(head)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(head)
        return distrax.Categorical(logits=logits)


class _StateDecoderHead(nn.Module):
    """Post-GRU state decoder: Dense -> activation -> Dense -> global state."""

    state_dim: int
    fc_dim_size: int
    activation: str

    @nn.compact
    def __call__(self, gru_out: jnp.ndarray) -> jnp.ndarray:
        activation = _activation_fn(self.activation)
        head = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(gru_out)
        head = activation(head)
        return nn.Dense(
            self.state_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(head)


class ActorDiscretePreAttnCommRNN(nn.Module):
    """
    Communicating discrete policy: attention before GRU.

    concat(obs, hidden_t) -> Dense -> message -> neighbor attention
    -> concat(message, context) -> GRU -> hidden_{t+1} -> Dense -> Categorical.

    Use ``message`` / ``act`` via ``apply(..., method=...)`` in the training loop.
    """

    action_dim: int
    fc_dim_size: int = 64
    msg_dim: int = 64
    activation: str = "tanh"

    def setup(self):
        self.message_layer = _ObsHiddenMessageLayer(
            msg_dim=self.msg_dim,
            activation=self.activation,
        )
        self.attention = _MaskedNeighborAttention(self.msg_dim)
        self.gru = ScannedRNN()
        self.action_head = _ActorGRUActionHead(
            action_dim=self.action_dim,
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )

    def message(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.message_layer(obs, hidden)

    def act(
        self,
        hidden: jnp.ndarray,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, distrax.Distribution]:
        context = self.attention(own_msg, neighbor_msgs, neighbor_mask)
        gru_in = jnp.concatenate([own_msg, context], axis=-1)
        if gru_in.ndim == 2:
            gru_in = gru_in[jnp.newaxis, :, :]
            done_in = dones[jnp.newaxis, :] if dones.ndim == 1 else dones
        else:
            done_in = dones
        new_hidden, gru_out = self.gru(hidden, (gru_in, done_in))
        if gru_out.ndim == 3:
            gru_out = gru_out.squeeze(0)
        pi = self.action_head(gru_out)
        return new_hidden, pi

    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution]:
        obs, dones, neighbor_msgs, neighbor_mask = x
        obs_step = obs[0] if obs.ndim == 3 else obs
        done_step = dones[0] if dones.ndim == 2 else dones
        msg = self.message(hidden, obs_step)
        new_hidden, pi = self.act(hidden, msg, neighbor_msgs, neighbor_mask, done_step)
        return new_hidden, msg, pi


class ActorDiscretePreAttnCommStateRNN(nn.Module):
    """
    Pre-attention comm actor with auxiliary global-state reconstruction.

    Same flow as ``ActorDiscretePreAttnCommRNN``; ``act`` also returns a
    state prediction from the GRU output via ``state_decoder``.
    """

    action_dim: int
    state_dim: int
    fc_dim_size: int = 64
    msg_dim: int = 64
    activation: str = "tanh"

    def setup(self):
        self.message_layer = _ObsHiddenMessageLayer(
            msg_dim=self.msg_dim,
            activation=self.activation,
        )
        self.attention = _MaskedNeighborAttention(self.msg_dim)
        self.gru = ScannedRNN()
        self.action_head = _ActorGRUActionHead(
            action_dim=self.action_dim,
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )
        self.state_decoder = _StateDecoderHead(
            state_dim=self.state_dim,
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )

    def message(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.message_layer(obs, hidden)

    def act(
        self,
        hidden: jnp.ndarray,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        context = self.attention(own_msg, neighbor_msgs, neighbor_mask)
        gru_in = jnp.concatenate([own_msg, context], axis=-1)
        if gru_in.ndim == 2:
            gru_in = gru_in[jnp.newaxis, :, :]
            done_in = dones[jnp.newaxis, :] if dones.ndim == 1 else dones
        else:
            done_in = dones
        new_hidden, gru_out = self.gru(hidden, (gru_in, done_in))
        if gru_out.ndim == 3:
            gru_out = gru_out.squeeze(0)
        pi = self.action_head(gru_out)
        state_pred = self.state_decoder(gru_out)
        return new_hidden, pi, state_pred

    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        obs, dones, neighbor_msgs, neighbor_mask = x
        obs_step = obs[0] if obs.ndim == 3 else obs
        done_step = dones[0] if dones.ndim == 2 else dones
        msg = self.message(hidden, obs_step)
        new_hidden, pi, state_pred = self.act(
            hidden, msg, neighbor_msgs, neighbor_mask, done_step
        )
        return new_hidden, msg, pi, state_pred


class ActorDiscretePreAttnCommStateInRNN(nn.Module):
    """
    Pre-attention comm actor with state reconstruction fed into the policy head.

    Same as ``ActorDiscretePreAttnCommStateRNN``, but the action head consumes
    ``concat(gru_out, stop_gradient(state_decoder(gru_out)))`` so PPO does not
    backprop into the state decoder (trained only via reconstruction loss).
    """

    action_dim: int
    state_dim: int
    fc_dim_size: int = 64
    msg_dim: int = 64
    activation: str = "tanh"

    def setup(self):
        self.message_layer = _ObsHiddenMessageLayer(
            msg_dim=self.msg_dim,
            activation=self.activation,
        )
        self.attention = _MaskedNeighborAttention(self.msg_dim)
        self.gru = ScannedRNN()
        self.state_decoder = _StateDecoderHead(
            state_dim=self.state_dim,
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )
        self.action_head = _ActorGRUActionHead(
            action_dim=self.action_dim,
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )

    def message(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.message_layer(obs, hidden)

    def act(
        self,
        hidden: jnp.ndarray,
        own_msg: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        context = self.attention(own_msg, neighbor_msgs, neighbor_mask)
        gru_in = jnp.concatenate([own_msg, context], axis=-1)
        if gru_in.ndim == 2:
            gru_in = gru_in[jnp.newaxis, :, :]
            done_in = dones[jnp.newaxis, :] if dones.ndim == 1 else dones
        else:
            done_in = dones
        new_hidden, gru_out = self.gru(hidden, (gru_in, done_in))
        if gru_out.ndim == 3:
            gru_out = gru_out.squeeze(0)
        state_pred = self.state_decoder(gru_out)
        policy_in = jnp.concatenate(
            [gru_out, jax.lax.stop_gradient(state_pred)], axis=-1
        )
        pi = self.action_head(policy_in)
        return new_hidden, pi, state_pred

    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        obs, dones, neighbor_msgs, neighbor_mask = x
        obs_step = obs[0] if obs.ndim == 3 else obs
        done_step = dones[0] if dones.ndim == 2 else dones
        msg = self.message(hidden, obs_step)
        new_hidden, pi, state_pred = self.act(
            hidden, msg, neighbor_msgs, neighbor_mask, done_step
        )
        return new_hidden, msg, pi, state_pred


class ActorDiscreteCommRNN(nn.Module):
    """
    Communicating discrete policy: encoder produces a message; head consumes
    concat(own message, neighbor messages from agents with index +/- 1).

    Use ``encode`` / ``act`` via ``apply(..., method=...)`` in the training loop.
    """

    action_dim: int
    hidden_size: int = 64
    fc_dim_size: int = 64
    neighbor_msg_dim: int = 64
    activation: str = "tanh"

    def setup(self):
        self.encoder = _ActorGRUEncoder(
            fc_dim_size=self.fc_dim_size,
            activation=self.activation,
        )
        self.head = _ActorCommHead(
            action_dim=self.action_dim,
            fc_dim_size=self.fc_dim_size,
            neighbor_msg_dim=self.neighbor_msg_dim,
            activation=self.activation,
        )

    def encode(
        self,
        hidden: jnp.ndarray,
        obs: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder(hidden, obs, dones)

    def act(
        self,
        message: jnp.ndarray,
        neighbor_msgs: jnp.ndarray,
    ) -> distrax.Distribution:
        return self.head(message, neighbor_msgs)

    def __call__(
        self,
        hidden: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution]:
        obs, dones, neighbor_msgs = x
        hidden, message = self.encode(hidden, obs, dones)
        pi = self.act(message, neighbor_msgs)
        return hidden, message, pi
