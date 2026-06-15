"""Team-shared MAPPO with GRU actor on JaxMARL MPE."""

import datetime
from typing import Any, NamedTuple, Optional, Tuple

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from talk.environments.mpe.jaxmarl_adapter import (
    TeamSpec,
    build_mpe_env,
    critic_state_dim,
    team_critic_state_from_unmasked,
)
from talk.experiments.mpe.ppo_minibatch import (
    hidden_to_actor_major,
    mask_logits_actor_major,
    num_team_actors,
    reshape_actor_minibatches,
    slot_agent_ids,
    tensor_to_actor_major,
)
from talk.environments.mpe.rollout_viz import log_rollout_video_callback
from talk.networks.gru import ActorDiscreteRNN, CriticDiscreteRNN, ScannedRNN
from talk.utils.jax_utils import pytree_norm
from talk.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER: Optional[WandbMultiLogger] = None


class Transition(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class TeamTraj(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class RunnerState(NamedTuple):
    actor_train_states: tuple
    critic_train_states: tuple
    actor_hidden_states: tuple
    critic_hidden_states: tuple
    env_state: Any
    obs: FloatArray
    unmasked_obs: FloatArray
    done: BoolArray
    cumulative_return: FloatArray
    timesteps: IntArray
    update_step: int
    rng: PRNGKeyArray


class TeamUpdateState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    init_actor_h: FloatArray
    init_critic_h: FloatArray
    traj_batch: TeamTraj
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def _mask_team_logits(logits: FloatArray, team: TeamSpec, action_dims_py: list[int]) -> FloatArray:
    agent_dims = jnp.array([action_dims_py[i] for i in team.agent_indices], dtype=jnp.int32)
    if logits.ndim == 3:
        valid = jnp.arange(team.action_dim)[None, None, :] < agent_dims[:, None, None]
    else:
        valid = jnp.arange(team.action_dim)[None, None, None, :] < agent_dims[None, None, :, None]
    return jnp.where(valid, logits, -1e10)


def _actor_step(
    actor_apply,
    params,
    hidden: FloatArray,
    team_obs: FloatArray,
    done: BoolArray,
    n_team: int,
) -> Tuple[FloatArray, distrax.Distribution]:
    num_envs = team_obs.shape[0]
    obs_ae = team_obs.transpose(1, 0, 2)
    batch = n_team * num_envs
    done_flat = jnp.broadcast_to(done[jnp.newaxis, :], (n_team, num_envs)).reshape(batch)
    ac_in = (
        obs_ae.reshape(1, batch, team_obs.shape[-1]),
        done_flat.reshape(1, batch),
    )
    new_h, pi = actor_apply(params, hidden.reshape(batch, -1), ac_in)
    return new_h.reshape(n_team, num_envs, -1), pi


def _actor_trajectory(
    actor_apply,
    params,
    init_hs: FloatArray,
    team_obs: FloatArray,
    done: BoolArray,
    team: TeamSpec,
    action_dims_py: list[int],
    trajectory_scan_unroll: int,
    slot_agents: Optional[jnp.ndarray] = None,
) -> FloatArray:
    """Replay actor over T steps via module-internal ScannedRNN (JaxMARL MAPPO style)."""
    del trajectory_scan_unroll
    if team_obs.ndim == 4:
        n_team = len(team.agent_indices)
        num_steps, num_envs = team_obs.shape[0], team_obs.shape[1]
        batch = n_team * num_envs
        obs_seq = team_obs.transpose(0, 2, 1, 3).reshape(num_steps, batch, team_obs.shape[-1])
        done_seq = tensor_to_actor_major(done, n_team, num_envs)
        _, pi = actor_apply(
            params,
            init_hs.reshape(batch, -1),
            (obs_seq, done_seq),
        )
        logits = pi.logits.reshape(num_steps, n_team, num_envs, -1).transpose(0, 2, 1, 3)
        return _mask_team_logits(logits, team, action_dims_py)

    num_steps, batch = team_obs.shape[0], team_obs.shape[1]
    _, pi = actor_apply(params, init_hs, (team_obs, done))
    slot_agents = (
        slot_agents
        if slot_agents is not None
        else slot_agent_ids(len(team.agent_indices), batch // len(team.agent_indices))
    )
    return mask_logits_actor_major(
        pi.logits,
        slot_agents,
        team.action_dim,
        action_dims_py,
        team.agent_indices,
    )


def _critic_step(
    critic_apply,
    params,
    hidden: FloatArray,
    team_global_state: FloatArray,
    done: BoolArray,
    n_team: int,
) -> Tuple[FloatArray, FloatArray]:
    num_envs = team_global_state.shape[0]
    gs_ae = team_global_state.transpose(1, 0, 2)
    batch = n_team * num_envs
    done_flat = jnp.broadcast_to(done[:, None], (num_envs, n_team)).T.reshape(batch)
    cr_in = (
        gs_ae.reshape(1, batch, team_global_state.shape[-1]),
        done_flat.reshape(1, batch),
    )
    new_h, values = critic_apply(params, hidden.reshape(batch, -1), cr_in)
    return new_h.reshape(n_team, num_envs, -1), values.reshape(n_team, num_envs)


def _calculate_gae(
    traj: TeamTraj, last_values: FloatArray, gamma: float, gae_lambda: float
) -> Tuple[FloatArray, FloatArray]:
    values = traj.value
    rewards = traj.reward
    dones = jnp.broadcast_to(traj.new_done[:, :, None], values.shape)

    def gae_one_agent(last_v, val, rew, done):
        def step(carry, transition):
            gae, next_value = carry
            delta = transition[0] + gamma * next_value * (1 - transition[1]) - transition[2]
            gae = delta + gamma * gae_lambda * (1 - transition[1]) * gae
            return (gae, transition[2]), gae

        _, advantages = jax.lax.scan(
            step,
            (jnp.zeros_like(last_v), last_v),
            (rew, done, val),
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + val

    advantages, targets = jax.vmap(gae_one_agent, in_axes=(0, 2, 2, 2))(
        last_values, values, rewards, dones
    )
    return advantages.transpose(1, 2, 0), targets.transpose(1, 2, 0)


def _slice_team_traj(
    traj: Transition,
    indices: IntArray,
    num_agents: int,
    max_obs_dim: int,
) -> TeamTraj:
    env_done = traj.done[:, :, 0] if traj.done.ndim == 3 else traj.done
    env_new_done = traj.new_done[:, :, 0] if traj.new_done.ndim == 3 else traj.new_done
    return TeamTraj(
        obs=traj.obs[:, :, indices, :],
        global_state=team_critic_state_from_unmasked(
            traj.global_state,
            indices,
            num_agents,
            max_obs_dim,
        ),
        action=traj.action[:, :, indices],
        log_prob=traj.log_prob[:, :, indices],
        reward=traj.reward[:, :, indices],
        done=env_done,
        new_done=env_new_done,
        value=traj.value[:, :, indices],
    )


def make_train(config: dict):
    adapter = build_mpe_env(
        env_name=config["env_name"],
        env_kwargs=config.get("env_kwargs", {}),
        sight_range=config["sight_range"],
        num_agents=config.get("num_agents"),
    )
    num_agents = adapter.num_agents
    max_obs_dim = adapter.max_obs_dim
    num_envs = config["num_envs"]
    team_specs = [t for t in adapter.team_specs if t.trainable]
    team_indices = [jnp.array(t.agent_indices, dtype=jnp.int32) for t in team_specs]
    hidden_size = config["gru_hidden_size"]
    if hidden_size != config["fc_dim_size"]:
        raise ValueError(
            "gru_hidden_size must equal fc_dim_size (GRU carry dim matches embedding dim)"
        )
    trajectory_scan_unroll = int(config.get("trajectory_scan_unroll", 8))
    action_dims_py = adapter.action_dims_py
    team_state_dim = critic_state_dim(num_agents, max_obs_dim)
    for team in team_specs:
        if team.action_kind != "discrete":
            raise ValueError("mappo_gru only supports discrete MPE teams")

    config["num_actors"] = num_agents * num_envs

    log_rollout_videos = config.get("log_rollout_videos", True) and config.get("use_wandb", True)
    rollout_fractions = config.get("log_rollout_fractions", [0.0, 0.25, 0.5, 0.75, 1.0])
    n_updates = config["num_update_steps"]
    checkpoint_steps = jnp.array(
        sorted(
            {int(round(float(fraction) * max(n_updates - 1, 0))) for fraction in rollout_fractions}
        ),
        dtype=jnp.int32,
    )
    rollout_ctx = {
        "env_name": config["env_name"],
        "env_kwargs": dict(config.get("env_kwargs", {}) or {}),
        "sight_range": config["sight_range"],
        "num_agents": config.get("num_agents"),
        "activation": config["activation"],
        "fc_dim_size": config["fc_dim_size"],
        "rollout_length_multiplier": float(config.get("rollout_length_multiplier", 2.0)),
        "rollout_eval_seed": int(config.get("rollout_eval_seed", 42)),
        "num_update_steps": n_updates,
        "log_seed": 0,
    }

    def rollout_io_callback(should_log, exp_id, update_step, actor_params):
        log_rollout_video_callback(
            bool(should_log),
            int(exp_id),
            int(update_step),
            actor_params,
            rollout_ctx,
            LOGGER,
        )

    def train(rng: PRNGKeyArray, exp_id: int):
        def train_setup(rng: PRNGKeyArray):
            rng, _rng_reset = jax.random.split(rng)
            obs, unmasked_obs, _, env_state = jax.vmap(adapter.reset)(
                jax.random.split(_rng_reset, num_envs)
            )

            def make_lr_schedule(base_lr: float):
                if config["anneal_lr"]:

                    def linear_schedule(count):
                        return base_lr * (1.0 - (count // config["num_gradient_steps"]))

                    return linear_schedule
                return base_lr

            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=make_lr_schedule(config["lr_actor"]), eps=1e-5),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=make_lr_schedule(config["lr_critic"]), eps=1e-5),
            )

            actor_states, critic_states, actor_hs_init, critic_hs_init = [], [], [], []
            rng_local = rng
            for team in team_specs:
                n_team = len(team.agent_indices)
                rng_local, rng_actor, rng_critic = jax.random.split(rng_local, 3)
                actor = ActorDiscreteRNN(
                    action_dim=team.action_dim,
                    hidden_size=hidden_size,
                    fc_dim_size=config["fc_dim_size"],
                    activation=config["activation"],
                )
                critic = CriticDiscreteRNN(
                    hidden_size=hidden_size,
                    fc_dim_size=config["fc_dim_size"],
                    activation=config["activation"],
                )
                ac_init_h = ScannedRNN.initialize_carry(n_team * num_envs, hidden_size)
                ac_in = (
                    jnp.zeros((1, n_team * num_envs, team.obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, n_team * num_envs), dtype=bool),
                )
                cr_init_h = ScannedRNN.initialize_carry(n_team * num_envs, hidden_size)
                cr_in = (
                    jnp.zeros((1, n_team * num_envs, team_state_dim), dtype=jnp.float32),
                    jnp.zeros((1, n_team * num_envs), dtype=bool),
                )
                actor_states.append(
                    TrainState.create(
                        apply_fn=actor.apply,
                        params=actor.init(rng_actor, ac_init_h, ac_in),
                        tx=tx_actor,
                    )
                )
                critic_states.append(
                    TrainState.create(
                        apply_fn=critic.apply,
                        params=critic.init(rng_critic, cr_init_h, cr_in),
                        tx=tx_critic,
                    )
                )
                actor_hs_init.append(ac_init_h.reshape(n_team, num_envs, hidden_size))
                critic_hs_init.append(cr_init_h.reshape(n_team, num_envs, hidden_size))

            return (
                obs,
                unmasked_obs,
                env_state,
                tuple(actor_states),
                tuple(critic_states),
                tuple(actor_hs_init),
                tuple(critic_hs_init),
                rng_local,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            obs,
            unmasked_obs,
            env_state,
            actor_train_states,
            critic_train_states,
            actor_hidden_states,
            critic_hidden_states,
            rng,
        ) = train_setup(_rng_setup)

        def _ppo_update_team(
            team: TeamSpec,
            actor_ts: TrainState,
            critic_ts: TrainState,
            init_actor_hs: FloatArray,
            init_critic_hs: FloatArray,
            traj: TeamTraj,
            last_values: FloatArray,
            rng_u: PRNGKeyArray,
        ):
            advantages, targets = _calculate_gae(
                traj, last_values, config["gamma"], config["gae_lambda"]
            )
            update_state = TeamUpdateState(
                actor_train_state=actor_ts,
                critic_train_state=critic_ts,
                init_actor_h=init_actor_hs,
                init_critic_h=init_critic_hs,
                traj_batch=traj,
                advantages=advantages,
                targets=targets,
                rng=rng_u,
            )

            def _update_epoch(update_state: TeamUpdateState, _):
                traj_b = update_state.traj_batch
                adv, tgt = update_state.advantages, update_state.targets
                rng_e, a_ts, c_ts = (
                    update_state.rng,
                    update_state.actor_train_state,
                    update_state.critic_train_state,
                )
                init_a_h = update_state.init_actor_h
                init_c_h = update_state.init_critic_h

                n_team = len(team.agent_indices)
                init_a_flat = hidden_to_actor_major(init_a_h, n_team, num_envs)
                init_c_flat = hidden_to_actor_major(init_c_h, n_team, num_envs)
                traj_actor = TeamTraj(
                    obs=tensor_to_actor_major(traj_b.obs, n_team, num_envs),
                    global_state=tensor_to_actor_major(traj_b.global_state, n_team, num_envs),
                    action=tensor_to_actor_major(traj_b.action, n_team, num_envs),
                    log_prob=tensor_to_actor_major(traj_b.log_prob, n_team, num_envs),
                    reward=tensor_to_actor_major(traj_b.reward, n_team, num_envs),
                    done=tensor_to_actor_major(traj_b.done, n_team, num_envs),
                    new_done=tensor_to_actor_major(traj_b.new_done, n_team, num_envs),
                    value=tensor_to_actor_major(traj_b.value, n_team, num_envs),
                )
                adv_am = tensor_to_actor_major(adv, n_team, num_envs)
                tgt_am = tensor_to_actor_major(tgt, n_team, num_envs)
                slot_agents = slot_agent_ids(n_team, num_envs)
                rng_e, key = jax.random.split(rng_e)
                perm = jax.random.permutation(key, num_team_actors(n_team, num_envs))
                init_a_flat = jnp.take(init_a_flat, perm, axis=0)
                init_c_flat = jnp.take(init_c_flat, perm, axis=0)
                traj_am = jax.tree.map(lambda x: jnp.take(x, perm, axis=1), traj_actor)
                adv_am = jnp.take(adv_am, perm, axis=1)
                tgt_am = jnp.take(tgt_am, perm, axis=1)
                slot_agents = jnp.take(slot_agents, perm, axis=0)
                num_mb = config["num_minibatches"]
                mb_actor_h = reshape_actor_minibatches(init_a_flat, num_mb, time_axis=False)
                mb_critic_h = reshape_actor_minibatches(init_c_flat, num_mb, time_axis=False)
                minibatches = (
                    mb_actor_h,
                    mb_critic_h,
                    jax.tree.map(lambda x: reshape_actor_minibatches(x, num_mb), traj_am),
                    reshape_actor_minibatches(adv_am, num_mb),
                    reshape_actor_minibatches(tgt_am, num_mb),
                    reshape_actor_minibatches(slot_agents, num_mb, time_axis=False),
                )

                def _update_minibatch(carry, minibatch):
                    a_ts_i, c_ts_i = carry
                    h_a, h_c, traj_mb, adv_mb, tgt_mb, slot_agents_mb = minibatch

                    def _actor_loss(params, init_h, traj_mb, adv_mb, slot_agents_mb):
                        logits = _actor_trajectory(
                            a_ts_i.apply_fn,
                            params,
                            init_h,
                            traj_mb.obs,
                            traj_mb.done,
                            team,
                            action_dims_py,
                            trajectory_scan_unroll,
                            slot_agents=slot_agents_mb,
                        )
                        masked_pi = distrax.Categorical(logits=logits)
                        log_prob = masked_pi.log_prob(traj_mb.action)
                        logratio = log_prob - traj_mb.log_prob
                        ratio = jnp.exp(logratio)
                        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                        loss_1, loss_2 = ratio * adv_mb, (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * adv_mb
                        )
                        loss_actor = -jnp.minimum(loss_1, loss_2).mean()
                        entropy = masked_pi.entropy().mean()
                        loss = loss_actor - config["ent_coef"] * entropy
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio - 1) - logratio).mean(),
                        }

                    def _critic_loss(params, init_h, traj_mb, tgt_mb):
                        _, value = c_ts_i.apply_fn(
                            params,
                            init_h,
                            (traj_mb.global_state, traj_mb.done),
                        )
                        value = value.reshape(traj_mb.value.shape)
                        value_clipped = traj_mb.value + (value - traj_mb.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - tgt_mb),
                                jnp.square(value_clipped - tgt_mb),
                            ).mean()
                        )
                        return loss, {"value_loss": loss}

                    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                        _actor_loss, has_aux=True
                    )(a_ts_i.params, h_a, traj_mb, adv_mb, slot_agents_mb)
                    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                        _critic_loss, has_aux=True
                    )(c_ts_i.params, h_c, traj_mb, tgt_mb)
                    a_ts_i = a_ts_i.apply_gradients(grads=actor_grads)
                    c_ts_i = c_ts_i.apply_gradients(grads=critic_grads)
                    aux = {
                        **actor_aux,
                        **critic_aux,
                        "total_loss": actor_loss + critic_loss,
                        "grad_norm_actor": pytree_norm(actor_grads),
                        "grad_norm_critic": pytree_norm(critic_grads),
                    }
                    return (a_ts_i, c_ts_i), aux

                (a_ts, c_ts), loss_info = jax.lax.scan(_update_minibatch, (a_ts, c_ts), minibatches)
                return (
                    TeamUpdateState(
                        actor_train_state=a_ts,
                        critic_train_state=c_ts,
                        init_actor_h=init_a_h,
                        init_critic_h=init_c_h,
                        traj_batch=traj_b,
                        advantages=adv,
                        targets=tgt,
                        rng=rng_e,
                    ),
                    loss_info,
                )

            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["num_epochs"]
            )
            return (
                update_state.actor_train_state,
                update_state.critic_train_state,
                loss_info,
            )

        def _train_loop(runner_state: RunnerState, _):
            initial_timesteps = runner_state.timesteps
            init_actor_hs = runner_state.actor_hidden_states
            init_critic_hs = runner_state.critic_hidden_states

            def _env_step(runner_state: RunnerState, _):
                obs_a = runner_state.obs
                global_state = runner_state.unmasked_obs
                last_done = runner_state.done
                rng_step = runner_state.rng
                actor_states = runner_state.actor_train_states
                critic_states = runner_state.critic_train_states
                actions = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
                log_probs = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
                values = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
                actions_cont = jnp.zeros(
                    (num_envs, num_agents, adapter.max_action_dim), dtype=jnp.float32
                )
                new_actor_hs_all = []
                new_critic_hs_all = []

                rng_step, _rng_action = jax.random.split(rng_step)
                for team_idx, team in enumerate(team_specs):
                    idx = team_indices[team_idx]
                    n_team = len(team.agent_indices)
                    team_obs = obs_a[:, idx, : team.obs_dim]
                    new_hs, pi = _actor_step(
                        actor_states[team_idx].apply_fn,
                        actor_states[team_idx].params,
                        runner_state.actor_hidden_states[team_idx],
                        team_obs,
                        last_done,
                        n_team,
                    )
                    logits = _mask_team_logits(
                        pi.logits.reshape(n_team, num_envs, team.action_dim),
                        team,
                        action_dims_py,
                    )
                    team_global = team_critic_state_from_unmasked(
                        global_state,
                        idx,
                        num_agents,
                        max_obs_dim,
                    )
                    new_critic_hs, team_values = _critic_step(
                        critic_states[team_idx].apply_fn,
                        critic_states[team_idx].params,
                        runner_state.critic_hidden_states[team_idx],
                        team_global,
                        last_done,
                        n_team,
                    )
                    _rng_action, key_team = jax.random.split(_rng_action)
                    logits_flat = logits.reshape(n_team * num_envs, team.action_dim)
                    masked_pi = distrax.Categorical(logits=logits_flat)
                    team_action = masked_pi.sample(seed=key_team)
                    team_log_prob = masked_pi.log_prob(team_action)
                    actions = actions.at[:, idx].set(team_action.reshape(num_envs, n_team))
                    log_probs = log_probs.at[:, idx].set(
                        team_log_prob.reshape(num_envs, n_team)
                    )
                    values = values.at[:, idx].set(team_values.transpose(1, 0))
                    new_actor_hs_all.append(new_hs)
                    new_critic_hs_all.append(new_critic_hs)

                rng_step, _rng_env = jax.random.split(rng_step)
                step_keys = jax.random.split(_rng_env, num_envs)
                new_obs, new_unmasked, _, new_env_state, rewards, new_done = jax.vmap(
                    lambda k, s, ad, ac: adapter.step(k, s, ad, ac)[:6]
                )(step_keys, runner_state.env_state, actions, actions_cont)
                timesteps = jnp.where(new_done, 0, runner_state.timesteps + 1)
                transition = Transition(
                    obs=obs_a,
                    global_state=global_state,
                    action=actions,
                    log_prob=log_probs,
                    reward=rewards,
                    done=last_done,
                    new_done=new_done,
                    value=values,
                )
                return (
                    RunnerState(
                        actor_train_states=actor_states,
                        critic_train_states=critic_states,
                        actor_hidden_states=tuple(new_actor_hs_all),
                        critic_hidden_states=tuple(new_critic_hs_all),
                        env_state=new_env_state,
                        obs=new_obs,
                        unmasked_obs=new_unmasked,
                        done=new_done,
                        cumulative_return=runner_state.cumulative_return,
                        timesteps=timesteps,
                        update_step=runner_state.update_step,
                        rng=rng_step,
                    ),
                    transition,
                )

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["num_steps_per_env_per_update"],
            )
            last_values_all = []
            for team_idx, team in enumerate(team_specs):
                team_global = team_critic_state_from_unmasked(
                    runner_state.unmasked_obs,
                    team_indices[team_idx],
                    num_agents,
                    max_obs_dim,
                )
                _, last_vals = _critic_step(
                    runner_state.critic_train_states[team_idx].apply_fn,
                    runner_state.critic_train_states[team_idx].params,
                    runner_state.critic_hidden_states[team_idx],
                    team_global,
                    runner_state.done,
                    len(team.agent_indices),
                )
                last_values_all.append(last_vals)

            rng_update, *team_rngs = jax.random.split(runner_state.rng, len(team_specs) + 1)
            new_actor_states, new_critic_states, loss_infos = [], [], []
            for team_idx, team in enumerate(team_specs):
                new_a, new_c, loss_i = _ppo_update_team(
                    team,
                    runner_state.actor_train_states[team_idx],
                    runner_state.critic_train_states[team_idx],
                    init_actor_hs[team_idx],
                    init_critic_hs[team_idx],
                    _slice_team_traj(
                        traj_batch,
                        team_indices[team_idx],
                        num_agents,
                        max_obs_dim,
                    ),
                    last_values_all[team_idx],
                    team_rngs[team_idx],
                )
                new_actor_states.append(new_a)
                new_critic_states.append(new_c)
                loss_infos.append(loss_i)

            actor_train_states = tuple(new_actor_states)
            critic_train_states = tuple(new_critic_states)
            loss_info = jax.tree.map(lambda *xs: jnp.stack(xs).mean(axis=(0, 1, 2)), *loss_infos)

            reward = traj_batch.reward
            done = traj_batch.new_done
            team_reward = reward.sum(axis=-1)

            def _rollout_episode_metrics(carry, inputs):
                partial_return, timestep_carry = carry
                r, d = inputs
                new_return = partial_return + r
                new_len = timestep_carry + 1
                return_at_done = jnp.where(d, new_return, 0.0)
                len_at_done = jnp.where(d, new_len.astype(jnp.float32), 0.0)
                next_partial = jnp.where(d, jnp.zeros_like(partial_return), new_return)
                next_timestep = jnp.where(d, jnp.zeros_like(timestep_carry), new_len)
                return (next_partial, next_timestep), (return_at_done, len_at_done)

            (new_cumulative_return, _), (ret_at_done, len_at_done) = jax.lax.scan(
                _rollout_episode_metrics,
                (runner_state.cumulative_return, initial_timesteps),
                (team_reward, done),
            )
            num_episodes = done.sum()
            metric = {
                "update_step": runner_state.update_step,
                "env_step": runner_state.update_step
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "return": jnp.where(num_episodes > 0, ret_at_done.sum() / num_episodes, 0.0),
                "episode_length": jnp.where(
                    num_episodes > 0, len_at_done.sum() / num_episodes, 0.0
                ),
                "num_episodes_completed": num_episodes,
                "mean_reward_per_step": reward.mean(),
                "actor_loss": loss_info["actor_loss"].mean(),
                "value_loss": loss_info["value_loss"].mean(),
                "entropy": loss_info["entropy"].mean(),
                "approx_kl": loss_info["approx_kl"].mean(),
                "total_loss": loss_info["total_loss"].mean(),
                "grad_norm_actor": loss_info["grad_norm_actor"].mean(),
                "grad_norm_critic": loss_info["grad_norm_critic"].mean(),
            }

            def callback(exp_id, metric_dict):
                if LOGGER is not None:
                    LOGGER.log(
                        int(exp_id),
                        {k: np.array(v) for k, v in metric_dict.items()},
                        step=int(metric_dict["update_step"]),
                    )

            jax.experimental.io_callback(callback, None, exp_id, metric)
            if log_rollout_videos:
                log_now = jnp.logical_and(
                    exp_id == 0,
                    jnp.any(runner_state.update_step == checkpoint_steps),
                )
                jax.experimental.io_callback(
                    rollout_io_callback,
                    None,
                    log_now,
                    exp_id,
                    runner_state.update_step,
                    tuple(ts.params for ts in actor_train_states),
                )

            return (
                RunnerState(
                    actor_train_states=actor_train_states,
                    critic_train_states=critic_train_states,
                    actor_hidden_states=runner_state.actor_hidden_states,
                    critic_hidden_states=runner_state.critic_hidden_states,
                    env_state=runner_state.env_state,
                    obs=runner_state.obs,
                    unmasked_obs=runner_state.unmasked_obs,
                    done=runner_state.done,
                    cumulative_return=new_cumulative_return,
                    timesteps=runner_state.timesteps,
                    update_step=runner_state.update_step + 1,
                    rng=rng_update,
                ),
                metric,
            )

        rng, _train_rng = jax.random.split(rng)
        final_runner, _ = jax.lax.scan(
            _train_loop,
            RunnerState(
                actor_train_states=actor_train_states,
                critic_train_states=critic_train_states,
                actor_hidden_states=actor_hidden_states,
                critic_hidden_states=critic_hidden_states,
                env_state=env_state,
                obs=obs,
                unmasked_obs=unmasked_obs,
                done=jnp.zeros((num_envs,), dtype=jnp.bool_),
                cumulative_return=jnp.zeros((num_envs,), dtype=jnp.float32),
                timesteps=jnp.zeros((num_envs,), dtype=jnp.int32),
                update_step=0,
                rng=_train_rng,
            ),
            None,
            length=config["num_update_steps"],
        )
        return final_runner

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_mappo_gru")
def main(config):
    global LOGGER
    try:
        config = OmegaConf.to_container(config)
        config["num_update_steps"] = (
            config["total_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["num_gradient_steps"] = (
            config["num_update_steps"] * config["num_epochs"] * config["num_minibatches"]
        )
        LOGGER = WandbMultiLogger(
            project=config["project"],
            job_type=config["algorithm"] + config.get("custom_name", ""),
            group=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            config=config,
            mode="online" if config["use_wandb"] else "disabled",
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )
        rng = jax.random.PRNGKey(config["seed"])
        print("Compiling MAPPO GRU MPE...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(
            train_fn(jax.random.split(rng, config["num_seeds"]), jnp.arange(config["num_seeds"]))
        )
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
