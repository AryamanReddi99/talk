"""Host-side MPE eval rollouts and wandb video rendering."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.patches import Circle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from talk.environments.mpe.comm_utils import gather_neighbors_by_comm_range
from talk.environments.mpe.jaxmarl_adapter import TeamSpec, build_mpe_env
from talk.experiments.mpe.env_utils import ally_comm_reachability
from talk.networks.gru import (
    ActorDiscretePreAttnCommStateRNN,
    ActorDiscreteRNN,
    ScannedRNN,
)
from talk.networks.tarmac import ActorTarMACRNN
from talk.networks.talk_decoder import ActorTalkRNN
from talk.networks.talk_codebook import ActorTalkCodebookRNN
from talk.networks.mordatch import ActorMordatchRNN
from talk.networks.mlp import ActorContinuous, ActorDiscrete

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def rollout_horizon_steps(adapter, length_multiplier):
    """Episode horizon in env steps (default MPE max_steps * multiplier)."""
    base_horizon = int(getattr(adapter.env, "max_steps", 25))
    return max(1, int(round(base_horizon * length_multiplier)))


def _make_actor(team, activation, fc_dim_size):
    if team.action_kind == "discrete":
        return ActorDiscrete(
            action_dim=team.action_dim, activation=activation, hidden_dim=fc_dim_size
        )
    return ActorContinuous(
        action_dim=team.action_dim, activation=activation, hidden_dim=fc_dim_size
    )


def _team_actions(team, params, obs, action_dims_py, activation, fc_dim_size):
    actor = _make_actor(team, activation, fc_dim_size)
    team_obs = obs[list(team.agent_indices), : team.obs_dim]
    n_team = len(team.agent_indices)
    if team.action_kind == "discrete":
        pi = actor.apply(params, team_obs)
        agent_dims = jnp.asarray(
            [action_dims_py[i] for i in team.agent_indices], dtype=jnp.int32
        )
        valid = jnp.arange(team.action_dim)[None, :] < agent_dims[:, None]
        masked_logits = jnp.where(valid, pi.logits, -1e10)
        actions = jnp.argmax(masked_logits, axis=-1)
        actions_d = actions.astype(jnp.int32)
        actions_c = jnp.zeros((n_team, team.action_dim), dtype=jnp.float32)
        return (actions_d, actions_c)
    pi = actor.apply(params, team_obs)
    actions_c = pi.loc.astype(jnp.float32)
    actions_d = jnp.zeros((n_team,), dtype=jnp.int32)
    return (actions_d, actions_c)


def _mask_team_logits(logits, team, action_dims_py):
    agent_dims = jnp.asarray(
        [action_dims_py[i] for i in team.agent_indices], dtype=jnp.int32
    )
    if logits.ndim == 2:
        valid = jnp.arange(team.action_dim)[None, :] < agent_dims[:, None]
    else:
        valid = jnp.arange(team.action_dim)[None, None, :] < agent_dims[:, None, None]
    return jnp.where(valid, logits, -1e10)


def run_gru_comm_eval_rollout(
    adapter,
    trainable_teams,
    actor_params,
    rng_key,
    max_steps,
    activation,
    fc_dim_size,
    gru_hidden_size,
    comm_range,
    max_neighbor_slots,
    stop_neighbor_msg_grad,
):
    """Single-env GRU comm rollout with zeroed hidden state at episode start."""
    key = rng_key
    (obs, unmasked, positions, state) = adapter.reset(key)
    state_seq = [state]
    total_reward = 0
    msg_dim = fc_dim_size
    actor_hs = [
        jnp.zeros((len(team.agent_indices), gru_hidden_size), dtype=jnp.float32)
        for team in trainable_teams
    ]
    for _ in range(max_steps):
        actions_d = jnp.zeros((adapter.num_agents,), dtype=jnp.int32)
        actions_c = jnp.zeros((adapter.num_agents, adapter.max_action_dim), dtype=jnp.float32)
        for team_idx, (team, params) in enumerate(zip(trainable_teams, actor_params)):
            idx = jnp.array(team.agent_indices, dtype=jnp.int32)
            n_team = len(team.agent_indices)
            team_obs = obs[idx, : team.obs_dim][None, :, :]
            team_positions = positions[idx, :][None, :, :]
            batch = n_team
            obs_ae = team_obs.transpose(1, 0, 2)
            actor = ActorDiscretePreAttnCommStateRNN(
                action_dim=team.action_dim,
                state_dim=team.obs_dim,
                fc_dim_size=fc_dim_size,
                msg_dim=msg_dim,
                activation=activation,
            )
            msgs = actor.apply(
                params,
                actor_hs[team_idx].reshape(batch, -1),
                obs_ae.reshape(batch, obs_ae.shape[-1]),
                method=ActorDiscretePreAttnCommStateRNN.message,
            )
            msgs = msgs.reshape(n_team, 1, msg_dim)
            if stop_neighbor_msg_grad:
                msgs = jax.lax.stop_gradient(msgs)
            (neighbor_msgs, neighbor_mask) = gather_neighbors_by_comm_range(
                msgs, team_positions, comm_range, max_neighbor_slots
            )
            done_flat = jnp.zeros((batch,), dtype=jnp.bool_)
            (new_hs, pi, _) = actor.apply(
                params,
                actor_hs[team_idx].reshape(batch, -1),
                msgs.reshape(batch, msg_dim),
                neighbor_msgs.transpose(0, 2, 1, 3).reshape(batch, max_neighbor_slots, msg_dim),
                neighbor_mask.transpose(0, 2, 1).reshape(batch, max_neighbor_slots),
                done_flat,
                method=ActorDiscretePreAttnCommStateRNN.act,
            )
            actor_hs[team_idx] = new_hs.reshape(n_team, gru_hidden_size)
            logits = _mask_team_logits(
                pi.logits.reshape(n_team, team.action_dim), team, adapter.action_dims_py
            )
            team_actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            for local_i, agent_idx in enumerate(team.agent_indices):
                actions_d = actions_d.at[agent_idx].set(team_actions[local_i])
        (key, key_step) = jax.random.split(key)
        (obs, unmasked, positions, state, rewards, done, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done):
            break
    return (state_seq, total_reward)


def run_eval_rollout(
    adapter, trainable_teams, actor_params, rng_key, max_steps, activation, fc_dim_size
):
    """Deterministic single-env rollout; returns states (incl. initial) and total reward."""
    key = rng_key
    (obs, unmasked, positions, state) = adapter.reset(key)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        actions_d = jnp.zeros((adapter.num_agents,), dtype=jnp.int32)
        actions_c = jnp.zeros((adapter.num_agents, adapter.max_action_dim), dtype=jnp.float32)
        for team, params in zip(trainable_teams, actor_params):
            (team_d, team_c) = _team_actions(
                team,
                params,
                obs,
                adapter.action_dims_py,
                activation=activation,
                fc_dim_size=fc_dim_size,
            )
            for local_i, agent_idx in enumerate(team.agent_indices):
                actions_d = actions_d.at[agent_idx].set(team_d[local_i])
                actions_c = actions_c.at[agent_idx, : team.action_dim].set(
                    team_c[local_i, : team.action_dim]
                )
        (key, key_step) = jax.random.split(key)
        (obs, unmasked, positions, state, rewards, done, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done):
            break
    return (state_seq, total_reward)


def run_gru_eval_rollout(
    adapter, actor_params, rng_key, max_steps, activation, fc_dim_size, gru_hidden_size
):
    """Deterministic single-env GRU rollout; returns states (incl. initial) and total reward."""
    params = (
        actor_params[0]
        if isinstance(actor_params, (tuple, list)) and len(actor_params) == 1
        else actor_params
    )
    action_dim = int(max(adapter.action_dims_py))
    obs_dim = int(adapter.max_obs_dim)
    num_agents = adapter.num_agents
    actor = ActorDiscreteRNN(
        action_dim=action_dim,
        hidden_size=gru_hidden_size,
        fc_dim_size=fc_dim_size,
        activation=activation,
    )
    key = rng_key
    (obs, _, _, state) = adapter.reset(key)
    hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
    done = jnp.zeros((num_agents,), dtype=bool)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        agent_obs = obs[:, :obs_dim]
        ac_in = (agent_obs[None, :, :], done[None, :])
        (hidden, pi) = actor.apply(params, hidden, ac_in)
        logits = pi.logits
        if logits.ndim == 3:
            logits = logits.squeeze(0)
        actions_d = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        actions_c = jnp.zeros((num_agents, adapter.max_action_dim), dtype=jnp.float32)
        (key, key_step) = jax.random.split(key)
        (obs, _, _, state, rewards, done_env, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        done = jnp.full((num_agents,), bool(done_env), dtype=bool)
        if bool(done_env):
            hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done_env):
            break
    return (state_seq, total_reward)


def run_tarmac_eval_rollout(
    adapter,
    actor_params,
    rng_key,
    max_steps,
    activation,
    fc_dim_size,
    gru_hidden_size,
    sig_dim,
    val_dim,
    comm_range,
):
    """Deterministic single-env TarMAC rollout; returns states (incl. initial) and total reward."""
    params = (
        actor_params[0]
        if isinstance(actor_params, (tuple, list)) and len(actor_params) == 1
        else actor_params
    )
    action_dim = int(max(adapter.action_dims_py))
    obs_dim = int(adapter.max_obs_dim)
    num_agents = adapter.num_agents
    actor = ActorTarMACRNN(
        action_dim=action_dim,
        hidden_size=gru_hidden_size,
        fc_dim_size=fc_dim_size,
        sig_dim=sig_dim,
        val_dim=val_dim,
        activation=activation,
    )
    key = rng_key
    (obs, _, positions, state) = adapter.reset(key)
    hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
    prev_sig = jnp.zeros((num_agents, sig_dim))
    prev_val = jnp.zeros((num_agents, val_dim))
    done = jnp.zeros((num_agents,), dtype=bool)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        reach = ally_comm_reachability(positions, comm_range)
        agent_obs = obs[:, :obs_dim]
        (hidden, sig, val, logits) = actor.apply(
            params,
            hidden,
            agent_obs,
            prev_sig,
            prev_val,
            done,
            reach,
            method=ActorTarMACRNN.step,
        )
        actions_d = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        actions_c = jnp.zeros((num_agents, adapter.max_action_dim), dtype=jnp.float32)
        (key, key_step) = jax.random.split(key)
        (obs, _, positions, state, rewards, done_env, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        done = jnp.full((num_agents,), bool(done_env), dtype=bool)
        if bool(done_env):
            hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
            prev_sig = jnp.zeros((num_agents, sig_dim))
            prev_val = jnp.zeros((num_agents, val_dim))
        else:
            prev_sig = sig
            prev_val = val
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done_env):
            break
    return (state_seq, total_reward)


def run_talk_codebook_eval_rollout(
    adapter,
    actor_params,
    rng_key,
    max_steps,
    activation,
    fc_dim_size,
    gru_hidden_size,
    sig_dim,
    codebook_size,
    vocab_dim,
    gumbel_tau,
    comm_range,
):
    """Receiver-side (deployment) single-env Talk-Codebook rollout.

    Comm uses each receiver's own codebook row at the received index
    (``receiver_lookup=True``); argmax token selection and argmax actions.
    """
    params = (
        actor_params[0]
        if isinstance(actor_params, (tuple, list)) and len(actor_params) == 1
        else actor_params
    )
    action_dim = int(max(adapter.action_dims_py))
    obs_dim = int(adapter.max_obs_dim)
    num_agents = adapter.num_agents
    actor = ActorTalkCodebookRNN(
        action_dim=action_dim,
        hidden_size=gru_hidden_size,
        fc_dim_size=fc_dim_size,
        sig_dim=sig_dim,
        codebook_size=codebook_size,
        vocab_dim=vocab_dim,
        gumbel_tau=gumbel_tau,
        activation=activation,
    )
    key = rng_key
    (obs, _, positions, state) = adapter.reset(key)
    hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
    prev_signature = jnp.zeros((num_agents, sig_dim))
    prev_codebook = jnp.zeros((num_agents, codebook_size, vocab_dim))
    prev_onehot = jnp.zeros((num_agents, codebook_size))
    done = jnp.zeros((num_agents,), dtype=bool)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        reach = ally_comm_reachability(positions, comm_range)
        agent_obs = obs[:, :obs_dim]
        (key, key_msg) = jax.random.split(key)
        msg_keys = jax.random.split(key_msg, num_agents)
        (hidden, sig, cb, onehot, logits, _aux, _comm, _diag) = actor.apply(
            params,
            hidden,
            agent_obs,
            prev_signature,
            prev_codebook,
            prev_onehot,
            done,
            msg_keys,
            reach,
            receiver_lookup=True,
            deterministic=True,
            method=ActorTalkCodebookRNN.step,
        )
        actions_d = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        actions_c = jnp.zeros((num_agents, adapter.max_action_dim), dtype=jnp.float32)
        (key, key_step) = jax.random.split(key)
        (obs, _, positions, state, rewards, done_env, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        done = jnp.full((num_agents,), bool(done_env), dtype=bool)
        if bool(done_env):
            hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
            prev_signature = jnp.zeros((num_agents, sig_dim))
            prev_codebook = jnp.zeros((num_agents, codebook_size, vocab_dim))
            prev_onehot = jnp.zeros((num_agents, codebook_size))
        else:
            prev_signature = sig
            prev_codebook = cb
            prev_onehot = onehot
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done_env):
            break
    return (state_seq, total_reward)


def log_mappo_talk_v2_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_talk_codebook_eval_rollout(
        adapter,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
        sig_dim=int(rollout_ctx["sig_dim"]),
        codebook_size=int(rollout_ctx["codebook_size"]),
        vocab_dim=int(rollout_ctx["vocab_dim"]),
        gumbel_tau=float(rollout_ctx["gumbel_tau"]),
        comm_range=float(rollout_ctx["comm_range"]),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def run_mordatch_eval_rollout(
    adapter,
    actor_params,
    rng_key,
    max_steps,
    activation,
    fc_dim_size,
    gru_hidden_size,
    vocab_size,
    msg_hidden_size,
    gumbel_tau,
    comm_range,
):
    """Deployment single-env Mordatch rollout; hard (argmax) messages and actions."""
    params = (
        actor_params[0]
        if isinstance(actor_params, (tuple, list)) and len(actor_params) == 1
        else actor_params
    )
    action_dim = int(max(adapter.action_dims_py))
    obs_dim = int(adapter.max_obs_dim)
    num_agents = adapter.num_agents
    actor = ActorMordatchRNN(
        action_dim=action_dim,
        obs_dim=obs_dim,
        hidden_size=gru_hidden_size,
        fc_dim_size=fc_dim_size,
        vocab_size=vocab_size,
        msg_hidden_size=msg_hidden_size,
        gumbel_tau=gumbel_tau,
        activation=activation,
    )
    key = rng_key
    (obs, _, positions, state) = adapter.reset(key)
    hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
    prev_msg = jnp.zeros((num_agents, vocab_size))
    prev_msg_mem = jnp.zeros((num_agents, num_agents, msg_hidden_size))
    done = jnp.zeros((num_agents,), dtype=bool)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        reach = ally_comm_reachability(positions, comm_range)
        agent_obs = obs[:, :obs_dim]
        (key, key_msg) = jax.random.split(key)
        msg_keys = jax.random.split(key_msg, num_agents)
        (hidden, new_msg, new_msg_mem, logits, _opl, _msp, _diag) = actor.apply(
            params,
            hidden,
            agent_obs,
            prev_msg,
            prev_msg_mem,
            done,
            msg_keys,
            reach,
            deterministic=True,
            method=ActorMordatchRNN.step,
        )
        actions_d = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        actions_c = jnp.zeros((num_agents, adapter.max_action_dim), dtype=jnp.float32)
        (key, key_step) = jax.random.split(key)
        (obs, _, positions, state, rewards, done_env, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        done = jnp.full((num_agents,), bool(done_env), dtype=bool)
        if bool(done_env):
            hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
            prev_msg = jnp.zeros((num_agents, vocab_size))
            prev_msg_mem = jnp.zeros((num_agents, num_agents, msg_hidden_size))
        else:
            prev_msg = new_msg
            prev_msg_mem = new_msg_mem
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done_env):
            break
    return (state_seq, total_reward)


def log_mordatch_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_mordatch_eval_rollout(
        adapter,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
        vocab_size=int(rollout_ctx["vocab_size"]),
        msg_hidden_size=int(rollout_ctx["msg_hidden_size"]),
        gumbel_tau=float(rollout_ctx["gumbel_tau"]),
        comm_range=float(rollout_ctx["comm_range"]),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def run_talk_eval_rollout(
    adapter,
    actor_params,
    rng_key,
    max_steps,
    activation,
    fc_dim_size,
    gru_hidden_size,
    vocab_content,
    vocab_embed_dim,
    attn_dim,
    max_msg_len,
    min_msg_len,
    gumbel_tau,
    comm_range,
):
    """Deterministic single-env Talk-Gumbel rollout; argmax tokens and actions."""
    params = (
        actor_params[0]
        if isinstance(actor_params, (tuple, list)) and len(actor_params) == 1
        else actor_params
    )
    action_dim = int(max(adapter.action_dims_py))
    obs_dim = int(adapter.max_obs_dim)
    num_agents = adapter.num_agents
    vocab_size = vocab_content + 2
    actor = ActorTalkRNN(
        action_dim=action_dim,
        hidden_size=gru_hidden_size,
        fc_dim_size=fc_dim_size,
        vocab_content=vocab_content,
        vocab_embed_dim=vocab_embed_dim,
        attn_dim=attn_dim,
        max_msg_len=max_msg_len,
        min_msg_len=min_msg_len,
        gumbel_tau=gumbel_tau,
        activation=activation,
    )
    key = rng_key
    (obs, _, positions, state) = adapter.reset(key)
    hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
    prev_tokens = jnp.zeros((num_agents, max_msg_len, vocab_size))
    prev_valid = jnp.zeros((num_agents, max_msg_len), dtype=bool)
    done = jnp.zeros((num_agents,), dtype=bool)
    state_seq = [state]
    total_reward = 0
    for _ in range(max_steps):
        reach = ally_comm_reachability(positions, comm_range)
        agent_obs = obs[:, :obs_dim]
        (key, key_msg) = jax.random.split(key)
        msg_keys = jax.random.split(key_msg, num_agents)
        (hidden, logits, msg_tokens, msg_valid, _, _) = actor.apply(
            params,
            hidden,
            agent_obs,
            prev_tokens,
            prev_valid,
            done,
            reach,
            msg_keys,
            deterministic=True,
            method=ActorTalkRNN.step,
        )
        actions_d = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        actions_c = jnp.zeros((num_agents, adapter.max_action_dim), dtype=jnp.float32)
        (key, key_step) = jax.random.split(key)
        (obs, _, positions, state, rewards, done_env, _) = adapter.step(
            key_step, state, actions_d, actions_c
        )
        done = jnp.full((num_agents,), bool(done_env), dtype=bool)
        if bool(done_env):
            hidden = ScannedRNN.initialize_carry(num_agents, gru_hidden_size)
            prev_tokens = jnp.zeros((num_agents, max_msg_len, vocab_size))
            prev_valid = jnp.zeros((num_agents, max_msg_len), dtype=bool)
        else:
            prev_tokens = msg_tokens
            prev_valid = msg_valid
        state_seq.append(state)
        total_reward += float(jnp.sum(rewards))
        if bool(done_env):
            break
    return (state_seq, total_reward)


def _fig_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def render_mpe_rollout_frames(env, state_seq, sight_range, comm_range=-1.0, fps=4):
    """Render top-down MPE frames as (T, H, W, 3) uint8."""
    del fps
    comm_active = not np.all(np.asarray(env.silent))
    (fig, ax) = plt.subplots(1, 1, figsize=(5, 5))
    ax_lim = 2
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_aspect("equal")
    ax.axis("off")
    sight_patches = []
    if sight_range >= 0:
        for _ in range(env.num_agents):
            patch = Circle(
                (0, 0),
                sight_range,
                facecolor="steelblue",
                edgecolor="steelblue",
                alpha=0.12,
                linewidth=0.8,
                linestyle="--",
                zorder=0,
            )
            ax.add_patch(patch)
            sight_patches.append(patch)
    comm_patches = []
    if comm_range >= 0:
        for _ in range(env.num_agents):
            patch = Circle(
                (0, 0),
                comm_range,
                facecolor="none",
                edgecolor="darkorange",
                alpha=1,
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
            ax.add_patch(patch)
            comm_patches.append(patch)
    entity_artists = []
    first_state = state_seq[0]
    for i in range(env.num_entities):
        pos = np.asarray(first_state.p_pos[i])
        entity = Circle(
            pos, float(env.rad[i]), color=np.asarray(env.colour[i]) / 255, zorder=2
        )
        ax.add_patch(entity)
        entity_artists.append(entity)
    step_counter = ax.text(-1.95, 1.95, "", va="top", fontsize=9, zorder=3)
    comm_artists = []
    comm_idx = np.array([], dtype=np.int32)
    if comm_active:
        comm_idx = np.where(np.asarray(env.silent) == 0)[0]
        for i, _idx in enumerate(comm_idx):
            comm_artists.append(
                ax.text(-1.95, -1.95 + i * 0.17, "", va="bottom", fontsize=8, zorder=3)
            )
    frames = []
    for state in state_seq:
        positions = np.asarray(state.p_pos)
        if sight_patches:
            for agent_i, patch in enumerate(sight_patches):
                patch.center = positions[agent_i]
        if comm_patches:
            for agent_i, patch in enumerate(comm_patches):
                patch.center = positions[agent_i]
        for i, entity in enumerate(entity_artists):
            entity.center = positions[i]
        step_counter.set_text(f"Step: {int(state.step)}")
        if comm_active:
            comm_values = np.asarray(state.c)
            for i, artist in enumerate(comm_artists):
                idx = int(comm_idx[i])
                letter = ALPHABET[int(np.argmax(comm_values[idx]))]
                artist.set_text(f"{env.agents[idx]} sends {letter}")
        frames.append(_fig_to_rgb(fig))
    plt.close(fig)
    return np.stack(frames, axis=0).astype(np.uint8)


def _frames_to_wandb_video(frames, fps):
    """Encode rollout frames for wandb without requiring moviepy."""
    import io
    import imageio.v2 as imageio

    bio = io.BytesIO()
    imageio.mimsave(bio, frames, format="GIF", duration=1000 / fps, loop=0)
    bio.seek(0)
    return wandb.Video(bio, format="gif", fps=fps)


def build_rollout_adapter(config):
    return build_mpe_env(
        env_name=config["env_name"],
        env_kwargs=config.get("env_kwargs", {}),
        sight_range=config["sight_range"],
        num_agents=config.get("num_agents"),
    )


def _scalar_io_callback_arg(value):
    """Coerce values passed through jax.experimental.io_callback to host scalars."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr


def log_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    trainable_teams = [t for t in adapter.team_specs if t.trainable]
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_eval_rollout(
        adapter,
        trainable_teams,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def log_mappo_tarmac_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_tarmac_eval_rollout(
        adapter,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
        sig_dim=int(rollout_ctx["sig_dim"]),
        val_dim=int(rollout_ctx["val_dim"]),
        comm_range=float(rollout_ctx["comm_range"]),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def log_mappo_talk_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_talk_eval_rollout(
        adapter,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
        vocab_content=int(rollout_ctx["vocab_content"]),
        vocab_embed_dim=int(rollout_ctx["vocab_embed_dim"]),
        attn_dim=int(rollout_ctx["attn_dim"]),
        max_msg_len=int(rollout_ctx["max_msg_len"]),
        min_msg_len=int(rollout_ctx.get("min_msg_len", 0)),
        gumbel_tau=float(rollout_ctx["gumbel_tau"]),
        comm_range=float(rollout_ctx["comm_range"]),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def log_mappo_gru_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    (state_seq, _) = run_gru_eval_rollout(
        adapter,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx.get("comm_range", -1)),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )


def log_gru_rollout_video_callback(
    should_log, exp_id, update_step, actor_params, rollout_ctx, logger, log_step=None
):
    should_log = bool(_scalar_io_callback_arg(should_log))
    exp_id = int(_scalar_io_callback_arg(exp_id))
    update_step = int(_scalar_io_callback_arg(update_step))
    if log_step is not None:
        log_step = int(_scalar_io_callback_arg(log_step))
    if not should_log or logger is None:
        return None
    if exp_id != int(rollout_ctx["log_seed"]):
        return None
    adapter = build_rollout_adapter(rollout_ctx)
    trainable_teams = [t for t in adapter.team_specs if t.trainable]
    max_steps = rollout_horizon_steps(adapter, rollout_ctx["rollout_length_multiplier"])
    eval_key = jax.random.PRNGKey(int(rollout_ctx["rollout_eval_seed"]))
    max_neighbor_slots = int(rollout_ctx["max_neighbor_slots"])
    (state_seq, _) = run_gru_comm_eval_rollout(
        adapter,
        trainable_teams,
        actor_params,
        eval_key,
        max_steps,
        activation=rollout_ctx["activation"],
        fc_dim_size=rollout_ctx["fc_dim_size"],
        gru_hidden_size=int(rollout_ctx["gru_hidden_size"]),
        comm_range=float(rollout_ctx["comm_range"]),
        max_neighbor_slots=max_neighbor_slots,
        stop_neighbor_msg_grad=bool(rollout_ctx.get("stop_neighbor_msg_grad", False)),
    )
    fraction = float(update_step) / max(1, int(rollout_ctx["num_update_steps"]) - 1)
    pct_label = f"{int(round(fraction * 100))}pct"
    frames = render_mpe_rollout_frames(
        adapter.env,
        state_seq,
        sight_range=float(rollout_ctx["sight_range"]),
        comm_range=float(rollout_ctx["comm_range"]),
    )
    video_key = f"rollout/{pct_label}"
    logger.log(
        int(rollout_ctx["log_seed"]),
        {video_key: _frames_to_wandb_video(frames, fps=4)},
        step=(log_step if log_step is not None else update_step),
    )
