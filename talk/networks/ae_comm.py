"""Simplified AEComm actor networks for FindGoal MAPPO."""

from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from talk.networks.gru import ScannedRNN
from talk.networks.mlp import _activation_fn


def straight_through_binarize(probs: jnp.ndarray) -> jnp.ndarray:
    """STE: forward hard 0/1, backward through sigmoid probabilities."""
    hard = (probs > 0.5).astype(probs.dtype)
    return probs + jax.lax.stop_gradient(hard - probs)


class ImgEncoder(nn.Module):
    """Two conv layers, adaptive pool to 3x3, FC to img_feat_dim."""

    img_feat_dim: int = 64
    activation: str = "elu"

    @nn.compact
    def __call__(self, pov: jnp.ndarray) -> jnp.ndarray:
        activation = _activation_fn(self.activation)
        x = pov
        if x.ndim == 3:
            x = x[jnp.newaxis, ...]
        x = nn.Conv(
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.img_feat_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        if pov.ndim == 3:
            x = x.squeeze(0)
        return x


class SelfposEncoder(nn.Module):
    """One-hot discrete (x, y) -> pos_feat_dim."""

    grid_size: int = 15
    pos_feat_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, selfpos: jnp.ndarray) -> jnp.ndarray:
        activation = _activation_fn(self.activation)
        x = selfpos.astype(jnp.int32)
        if x.ndim == 1:
            x = x[jnp.newaxis, ...]
        x_onehot = jnp.concatenate(
            [
                jax.nn.one_hot(x[:, 0], self.grid_size),
                jax.nn.one_hot(x[:, 1], self.grid_size),
            ],
            axis=-1,
        )
        x = nn.Dense(
            self.pos_feat_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x_onehot)
        x = activation(x)
        if selfpos.ndim == 1:
            x = x.squeeze(0)
        return x


class ObsFeatEncoder(nn.Module):
    """POV + selfpos -> feat_dim (img + pos)."""

    grid_size: int = 15
    img_feat_dim: int = 64
    pos_feat_dim: int = 64
    activation: str = "tanh"

    @property
    def feat_dim(self) -> int:
        return self.img_feat_dim + self.pos_feat_dim

    @nn.compact
    def __call__(self, pov: jnp.ndarray, selfpos: jnp.ndarray) -> jnp.ndarray:
        img = ImgEncoder(
            img_feat_dim=self.img_feat_dim,
            activation="elu",
        )(pov)
        pos = SelfposEncoder(
            grid_size=self.grid_size,
            pos_feat_dim=self.pos_feat_dim,
            activation=self.activation,
        )(selfpos)
        return jnp.concatenate([img, pos], axis=-1)


class CommAutoencoder(nn.Module):
    """Single latent layer: feat -> comm_len -> feat reconstruction."""

    feat_dim: int
    comm_len: int = 10

    def setup(self):
        self.encoder = nn.Dense(
            self.comm_len,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.decoder = nn.Dense(
            self.feat_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

    def decode(self, msg: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(msg)

    def __call__(self, feat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        feat_sg = jax.lax.stop_gradient(feat)
        probs = jax.nn.sigmoid(self.encoder(feat_sg))
        msg = straight_through_binarize(probs)
        recon = self.decoder(msg)
        ae_loss = jnp.mean(jnp.square(recon - feat_sg), axis=-1)
        return jax.lax.stop_gradient(msg), ae_loss


class AECommActor(nn.Module):
    """AEComm actor: encode obs, AE message, decode others, GRU policy head."""

    action_dim: int = 5
    num_agents: int = 3
    grid_size: int = 15
    img_feat_dim: int = 64
    pos_feat_dim: int = 64
    comm_len: int = 10
    hidden_size: int = 256
    fc_dim_size: int = 64
    activation: str = "tanh"

    @property
    def feat_dim(self) -> int:
        return self.img_feat_dim + self.pos_feat_dim

    @property
    def listener_dim(self) -> int:
        return self.feat_dim + (self.num_agents - 1) * self.feat_dim + self.comm_len

    def setup(self):
        self.obs_encoder = ObsFeatEncoder(
            grid_size=self.grid_size,
            img_feat_dim=self.img_feat_dim,
            pos_feat_dim=self.pos_feat_dim,
            activation=self.activation,
        )
        self.autoencoder = CommAutoencoder(
            feat_dim=self.feat_dim,
            comm_len=self.comm_len,
        )
        self.gru = ScannedRNN()
        self.listener_proj = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.policy_fc = nn.Dense(
            self.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )
        self.action_head = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )

    def encode_obs(self, pov: jnp.ndarray, selfpos: jnp.ndarray) -> jnp.ndarray:
        return self.obs_encoder(pov, selfpos)

    def encode_message(self, feat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.autoencoder(feat)

    def decode_messages(self, msgs: jnp.ndarray) -> jnp.ndarray:
        return self.autoencoder.decode(msgs)

    def listener_input(
        self,
        feat: jnp.ndarray,
        prev_msgs: jnp.ndarray,
        msg: jnp.ndarray,
        agent_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        if prev_msgs.ndim == 2:
            prev_msgs = prev_msgs[jnp.newaxis, ...]
            feat = feat[jnp.newaxis, ...]
            msg = msg[jnp.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        other_indices = jnp.array(
            [[1, 2], [0, 2], [0, 1]][: self.num_agents],
            dtype=jnp.int32,
        )
        agent_idx_arr = jnp.broadcast_to(
            jnp.atleast_1d(agent_idx), (prev_msgs.shape[0],)
        )
        batch_idx = jnp.arange(prev_msgs.shape[0])[:, None]
        others = prev_msgs[batch_idx, other_indices[agent_idx_arr], :]
        decoded = self.decode_messages(others)
        decoded_flat = decoded.reshape(decoded.shape[0], -1)
        out = jnp.concatenate(
            [
                feat,
                jax.lax.stop_gradient(decoded_flat),
                jax.lax.stop_gradient(msg),
            ],
            axis=-1,
        )
        if squeeze:
            out = out.squeeze(0)
        return out

    def act(
        self,
        hidden: jnp.ndarray,
        pov: jnp.ndarray,
        selfpos: jnp.ndarray,
        prev_msgs: jnp.ndarray,
        msg: jnp.ndarray,
        agent_idx: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        """Batched act for all (agent, env) pairs with shared weights."""
        feat = self.encode_obs(pov, selfpos)
        gru_in = self.listener_proj(self.listener_input(feat, prev_msgs, msg, agent_idx))
        if gru_in.ndim == 2:
            gru_in = gru_in[jnp.newaxis, :, :]
            done_in = dones[jnp.newaxis, :] if dones.ndim == 1 else dones
        else:
            done_in = dones
        new_hidden, gru_out = self.gru(hidden, (gru_in, done_in))
        if gru_out.ndim == 3:
            gru_out = gru_out.squeeze(0)
        activation = _activation_fn(self.activation)
        head = activation(self.policy_fc(gru_out))
        logits = self.action_head(head)
        _, ae_loss = self.encode_message(feat)
        return new_hidden, distrax.Categorical(logits=logits), ae_loss

    def __call__(
        self,
        hidden: jnp.ndarray,
        pov: jnp.ndarray,
        selfpos: jnp.ndarray,
        prev_msgs: jnp.ndarray,
        agent_idx: int,
        dones: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Distribution, jnp.ndarray]:
        feat = self.encode_obs(pov, selfpos)
        msg, ae_loss = self.encode_message(feat)
        new_hidden, pi, _ = self.act(
            hidden, pov, selfpos, prev_msgs, msg, agent_idx, dones
        )
        return new_hidden, msg, pi, ae_loss
