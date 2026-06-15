"""Team-shared MAPPO with pre-attention GRU comm + state reconstruction on JaxMARL MPE."""

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

from talk.environments.mpe.comm_utils import (
    gather_neighbors_actor_major,
    gather_neighbors_by_comm_range,
    max_neighbor_slots as compute_max_neighbor_slots,
)
from talk.environments.mpe.jaxmarl_adapter import (
    TeamSpec,
    build_mpe_env,
    critic_state_dim,
    team_critic_state_from_unmasked,
)
from talk.environments.mpe.rollout_viz import log_gru_rollout_video_callback
from talk.experiments.mpe.ppo_minibatch import (
    hidden_to_actor_major,
    mask_logits_actor_major,
    num_team_actors,
    reshape_actor_minibatches,
    slot_agent_ids,
    slot_env_ids,
    tensor_to_actor_major,
)
from talk.networks.gru import (
    ActorDiscretePreAttnCommStateRNN,
    ScannedRNN,
)
from talk.networks.mlp import CriticDiscrete
from talk.utils.jax_utils import pytree_norm
from talk.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER: Optional[WandbMultiLogger] = None


class Transition(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    positions: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class TeamTraj(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    positions: FloatArray
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
    env_state: Any
    obs: FloatArray
    unmasked_obs: FloatArray
    positions: FloatArray
    done: BoolArray
    cumulative_return: FloatArray
    timesteps: IntArray
    update_step: int
    rng: PRNGKeyArray


class TeamUpdateState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    init_actor_h: FloatArray
    traj_batch: TeamTraj
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def _message_step(
    actor_apply,
    params,
    actor_hs: FloatArray,
    team_obs: FloatArray,
    n_team: int,
) -> FloatArray:
    """team_obs: (E, n_team, D) -> msgs (n_team, E, msg_dim)."""
    num_envs = team_obs.shape[0]
    obs_ae = team_obs.transpose(1, 0, 2)
    batch = n_team * num_envs
    msgs = actor_apply(
        params,
        actor_hs.reshape(batch, -1),
        obs_ae.reshape(batch, obs_ae.shape[-1]),
        method=ActorDiscretePreAttnCommStateRNN.message,
    )
    msg_dim = msgs.shape[-1]
    return msgs.reshape(n_team, num_envs, msg_dim)


def _message_step_flat(
    actor_apply,
    params,
    actor_hs: FloatArray,
    obs: FloatArray,
) -> FloatArray:
    """obs: (batch, D) -> msgs (batch, msg_dim)."""
    return actor_apply(
        params,
        actor_hs,
        obs,
        method=ActorDiscretePreAttnCommStateRNN.message,
    )


def _act_batched(
    actor_apply,
    params,
    hidden: FloatArray,
    own_msgs: FloatArray,
    neighbor_msgs: FloatArray,
    neighbor_mask: FloatArray,
    dones: FloatArray,
) -> Tuple[FloatArray, distrax.Distribution, FloatArray]:
    return actor_apply(
        params,
        hidden,
        own_msgs,
        neighbor_msgs,
        neighbor_mask,
        dones,
        method=ActorDiscretePreAttnCommStateRNN.act,
    )


def _mask_team_logits(
    logits: FloatArray, team: TeamSpec, action_dims_py: list[int]
) -> FloatArray:
    agent_dims = jnp.array(
        [action_dims_py[i] for i in team.agent_indices], dtype=jnp.int32
    )
    if logits.ndim == 3:
        valid = jnp.arange(team.action_dim)[None, None, :] < agent_dims[:, None, None]
    else:
        valid = (
            jnp.arange(team.action_dim)[None, None, None, :]
            < agent_dims[None, None, :, None]
        )
    return jnp.where(valid, logits, -1e10)


def _actor_trajectory(
    actor_apply,
    params,
    init_hs: FloatArray,
    team_obs: FloatArray,
    team_positions: FloatArray,
    done: BoolArray,
    team: TeamSpec,
    action_dims_py: list[int],
    comm_range: float,
    max_neighbor_slots: int,
    stop_neighbor_msg_grad: bool = False,
    trajectory_scan_unroll: int = 8,
    slot_agents: Optional[jnp.ndarray] = None,
) -> Tuple[FloatArray, FloatArray]:
    n_team = len(team.agent_indices)

    if team_obs.ndim == 3:
        num_envs = team_obs.shape[1] // n_team
        env_ids = slot_env_ids(n_team, num_envs)
        slot_agents = (
            slot_agents
            if slot_agents is not None
            else slot_agent_ids(n_team, num_envs)
        )

        def step_flat(h, inputs):
            obs_t, pos_t, done_t = inputs
            msgs = _message_step_flat(actor_apply, params, h, obs_t)
            if stop_neighbor_msg_grad:
                msgs = jax.lax.stop_gradient(msgs)
            neighbor_msgs, neighbor_mask = gather_neighbors_actor_major(
                msgs,
                pos_t,
                slot_agents,
                env_ids,
                comm_range,
                max_neighbor_slots,
                n_team,
            )
            new_h, pi, state_pred = _act_batched(
                actor_apply,
                params,
                h,
                msgs,
                neighbor_msgs,
                neighbor_mask,
                done_t,
            )
            return new_h, (pi.logits, state_pred)

        _, (logits, state_preds) = jax.lax.scan(
            step_flat,
            init_hs,
            (team_obs, team_positions, done),
            unroll=trajectory_scan_unroll,
        )
        logits = mask_logits_actor_major(
            logits,
            slot_agents,
            team.action_dim,
            action_dims_py,
            team.agent_indices,
        )
        return logits, state_preds

    def step(h, inputs):
        obs_t, pos_t, done_t = inputs
        msgs = _message_step(actor_apply, params, h, obs_t, n_team)
        if stop_neighbor_msg_grad:
            msgs = jax.lax.stop_gradient(msgs)
        neighbor_msgs, neighbor_mask = gather_neighbors_by_comm_range(
            msgs,
            pos_t,
            comm_range,
            max_neighbor_slots,
        )
        batch = n_team * obs_t.shape[0]
        done_flat = jnp.broadcast_to(
            done_t[jnp.newaxis, :], (n_team, obs_t.shape[0])
        ).reshape(batch)
        new_h, pi, state_pred = _act_batched(
            actor_apply,
            params,
            h.reshape(batch, -1),
            msgs.reshape(batch, -1),
            neighbor_msgs.transpose(0, 2, 1, 3).reshape(
                batch, max_neighbor_slots, msgs.shape[-1]
            ),
            neighbor_mask.transpose(0, 2, 1).reshape(batch, max_neighbor_slots),
            done_flat,
        )
        logits = pi.logits.reshape(n_team, obs_t.shape[0], -1)
        state_pred = state_pred.reshape(n_team, obs_t.shape[0], -1)
        return new_h.reshape(n_team, obs_t.shape[0], -1), (logits, state_pred)

    _, (logits, state_preds) = jax.lax.scan(
        step,
        init_hs,
        (team_obs, team_positions, done),
        unroll=trajectory_scan_unroll,
    )
    logits = logits.transpose(0, 2, 1, 3)
    state_preds = state_preds.transpose(0, 2, 1, 3)
    logits = _mask_team_logits(logits, team, action_dims_py)
    return logits, state_preds


def _team_critic_values(
    critic_apply,
    params,
    team_global_state: FloatArray,
    n_team: int,
) -> FloatArray:
    num_envs = team_global_state.shape[0]
    gs = team_global_state.reshape(-1, team_global_state.shape[-1])
    values = critic_apply(params, gs)
    return values.reshape(num_envs, n_team).transpose(1, 0)


def _calculate_gae_batched(
    traj: TeamTraj,
    last_values: FloatArray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[FloatArray, FloatArray]:
    values = traj.value
    rewards = traj.reward
    dones = jnp.broadcast_to(traj.new_done[:, :, None], values.shape)

    def gae_one_agent(last_v, val, rew, done):
        def step(carry, transition):
            gae, next_value = carry
            delta = (
                transition[0] + gamma * next_value * (1 - transition[1]) - transition[2]
            )
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
    if traj.done.ndim == 3:
        env_done = traj.done[:, :, 0]
        env_new_done = traj.new_done[:, :, 0]
    else:
        env_done = traj.done
        env_new_done = traj.new_done
    return TeamTraj(
        obs=traj.obs[:, :, indices, :],
        global_state=team_critic_state_from_unmasked(
            traj.global_state,
            indices,
            num_agents,
            max_obs_dim,
        ),
        positions=traj.positions[:, :, indices, :],
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
    max_neighbor_slots = compute_max_neighbor_slots(team_specs)
    comm_range = float(config["comm_range"])
    msg_dim = config["fc_dim_size"]
    hidden_size = config["gru_hidden_size"]
    stop_neighbor_msg_grad = config.get("stop_neighbor_msg_grad", False)
    trajectory_scan_unroll = int(config.get("trajectory_scan_unroll", 8))
    action_dims_py = adapter.action_dims_py
    team_state_dim = critic_state_dim(num_agents, max_obs_dim)
    for team in team_specs:
        if team.action_kind != "discrete":
            raise ValueError(
                "mappo_att_grucomm_dub_state only supports discrete MPE teams; "
                f"team {team.name} has action_kind={team.action_kind}"
            )

    config["num_actors"] = num_agents * num_envs

    log_rollout_videos = config.get("log_rollout_videos", True) and config.get(
        "use_wandb", True
    )
    rollout_fractions = config.get("log_rollout_fractions", [0.0, 0.25, 0.5, 0.75, 1.0])
    n_updates = config["num_update_steps"]
    checkpoint_steps = jnp.array(
        sorted(
            {
                int(round(float(fraction) * max(n_updates - 1, 0)))
                for fraction in rollout_fractions
            }
        ),
        dtype=jnp.int32,
    )
    rollout_ctx = {
        "env_name": config["env_name"],
        "env_kwargs": dict(config.get("env_kwargs", {}) or {}),
        "sight_range": config["sight_range"],
        "comm_range": config["comm_range"],
        "num_agents": config.get("num_agents"),
        "activation": config["activation"],
        "fc_dim_size": config["fc_dim_size"],
        "gru_hidden_size": config["gru_hidden_size"],
        "max_neighbor_slots": max_neighbor_slots,
        "stop_neighbor_msg_grad": stop_neighbor_msg_grad,
        "rollout_length_multiplier": float(
            config.get("rollout_length_multiplier", 2.0)
        ),
        "rollout_eval_seed": int(config.get("rollout_eval_seed", 42)),
        "num_update_steps": n_updates,
        "log_seed": 0,
    }

    def rollout_io_callback(should_log, exp_id, update_step, actor_params):
        log_gru_rollout_video_callback(
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
            reset_keys = jax.random.split(_rng_reset, num_envs)
            obs, unmasked_obs, positions, env_state = jax.vmap(adapter.reset)(
                reset_keys
            )

            def make_lr_schedule(base_lr: float):
                if config["anneal_lr"]:

                    def linear_schedule(count):
                        frac = 1.0 - (count // config["num_gradient_steps"])
                        return base_lr * frac

                    return linear_schedule
                return base_lr

            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(
                    learning_rate=make_lr_schedule(config["lr_actor"]), eps=1e-5
                ),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(
                    learning_rate=make_lr_schedule(config["lr_critic"]), eps=1e-5
                ),
            )

            actor_states = []
            critic_states = []
            actor_hs_init = []
            rng_local = rng
            for team in team_specs:
                n_team = len(team.agent_indices)
                rng_local, rng_actor, rng_critic = jax.random.split(rng_local, 3)
                actor = ActorDiscretePreAttnCommStateRNN(
                    action_dim=team.action_dim,
                    state_dim=team_state_dim,
                    fc_dim_size=config["fc_dim_size"],
                    msg_dim=msg_dim,
                    activation=config["activation"],
                )
                critic = CriticDiscrete(
                    activation=config["activation"],
                    hidden_dim=config["fc_dim_size"],
                )
                ac_init_h = ScannedRNN.initialize_carry(n_team * num_envs, hidden_size)
                ac_in = (
                    jnp.zeros((1, n_team * num_envs, team.obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, n_team * num_envs), dtype=bool),
                    jnp.zeros(
                        (n_team * num_envs, max_neighbor_slots, msg_dim),
                        dtype=jnp.float32,
                    ),
                    jnp.zeros(
                        (n_team * num_envs, max_neighbor_slots), dtype=jnp.float32
                    ),
                )
                dummy_global = jnp.zeros((team_state_dim,), dtype=jnp.float32)
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
                        params=critic.init(rng_critic, dummy_global),
                        tx=tx_critic,
                    )
                )
                actor_hs_init.append(ac_init_h.reshape(n_team, num_envs, hidden_size))

            return (
                obs,
                unmasked_obs,
                positions,
                env_state,
                tuple(actor_states),
                tuple(critic_states),
                tuple(actor_hs_init),
                rng_local,
            )

        rng, _rng_setup = jax.random.split(rng)
        (
            obs,
            unmasked_obs,
            positions,
            env_state,
            actor_train_states,
            critic_train_states,
            actor_hidden_states,
            rng,
        ) = train_setup(_rng_setup)

        def _ppo_update_team(
            team: TeamSpec,
            actor_ts: TrainState,
            critic_ts: TrainState,
            init_actor_hs: FloatArray,
            traj: TeamTraj,
            last_values: FloatArray,
            rng_u: PRNGKeyArray,
        ):
            advantages, targets = _calculate_gae_batched(
                traj, last_values, config["gamma"], config["gae_lambda"]
            )
            update_state = TeamUpdateState(
                actor_train_state=actor_ts,
                critic_train_state=critic_ts,
                init_actor_h=init_actor_hs,
                traj_batch=traj,
                advantages=advantages,
                targets=targets,
                rng=rng_u,
            )

            def _update_epoch(update_state: TeamUpdateState, _):
                traj_b = update_state.traj_batch
                adv = update_state.advantages
                tgt = update_state.targets
                rng_e = update_state.rng
                a_ts = update_state.actor_train_state
                c_ts = update_state.critic_train_state
                init_a_h = update_state.init_actor_h

                n_team = len(team.agent_indices)
                init_flat = hidden_to_actor_major(init_a_h, n_team, num_envs)
                traj_actor = TeamTraj(
                    obs=tensor_to_actor_major(traj_b.obs, n_team, num_envs),
                    global_state=tensor_to_actor_major(
                        traj_b.global_state, n_team, num_envs
                    ),
                    positions=tensor_to_actor_major(traj_b.positions, n_team, num_envs),
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
                init_flat = jnp.take(init_flat, perm, axis=0)
                traj_am = jax.tree.map(
                    lambda x: jnp.take(x, perm, axis=1), traj_actor
                )
                adv_am = jnp.take(adv_am, perm, axis=1)
                tgt_am = jnp.take(tgt_am, perm, axis=1)
                slot_agents = jnp.take(slot_agents, perm, axis=0)
                num_mb = config["num_minibatches"]
                mb_actor_h = reshape_actor_minibatches(
                    init_flat, num_mb, time_axis=False
                )
                minibatches = (
                    mb_actor_h,
                    jax.tree.map(
                        lambda x: reshape_actor_minibatches(x, num_mb), traj_am
                    ),
                    reshape_actor_minibatches(adv_am, num_mb),
                    reshape_actor_minibatches(tgt_am, num_mb),
                    reshape_actor_minibatches(slot_agents, num_mb, time_axis=False),
                )

                def _update_minibatch(carry, minibatch):
                    a_ts_i, c_ts_i = carry
                    h_a, traj_mb, adv_mb, tgt_mb, slot_agents_mb = minibatch

                    def _actor_loss(params, init_h, traj_mb, adv_mb, slot_agents_mb):
                        logits, state_preds = _actor_trajectory(
                            a_ts_i.apply_fn,
                            params,
                            init_h,
                            traj_mb.obs,
                            traj_mb.positions,
                            traj_mb.done,
                            team,
                            action_dims_py,
                            comm_range,
                            max_neighbor_slots,
                            stop_neighbor_msg_grad,
                            trajectory_scan_unroll,
                            slot_agents=slot_agents_mb,
                        )
                        masked_pi = distrax.Categorical(logits=logits)
                        log_prob = masked_pi.log_prob(traj_mb.action)
                        logratio = log_prob - traj_mb.log_prob
                        ratio = jnp.exp(logratio)
                        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                        loss_1 = ratio * adv_mb
                        loss_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * adv_mb
                        )
                        loss_actor = -jnp.minimum(loss_1, loss_2).mean()
                        entropy = masked_pi.entropy().mean()
                        logratio_flat = logratio.reshape(-1)
                        ratio_flat = ratio.reshape(-1)
                        state_target = jax.lax.stop_gradient(traj_mb.global_state)
                        state_mse = jnp.square(state_preds - state_target).mean()
                        loss = (
                            loss_actor
                            - config["ent_coef"] * entropy
                            + config["recon_coef"] * state_mse
                        )
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio_flat - 1) - logratio_flat).mean(),
                            "state_mse": state_mse,
                        }

                    def _critic_loss(params, traj_mb, tgt_mb):
                        gs = traj_mb.global_state.reshape(-1, team_state_dim)
                        value = c_ts_i.apply_fn(params, gs).reshape(traj_mb.value.shape)
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
                    )(c_ts_i.params, traj_mb, tgt_mb)

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

                (a_ts, c_ts), loss_info = jax.lax.scan(
                    _update_minibatch,
                    (a_ts, c_ts),
                    minibatches,
                )
                return (
                    TeamUpdateState(
                        actor_train_state=a_ts,
                        critic_train_state=c_ts,
                        init_actor_h=init_a_h,
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

            def _env_step(runner_state: RunnerState, _):
                actor_states = runner_state.actor_train_states
                critic_states = runner_state.critic_train_states
                actor_hs = runner_state.actor_hidden_states
                obs_a = runner_state.obs
                env_state = runner_state.env_state
                global_state = runner_state.unmasked_obs
                positions = runner_state.positions
                last_done = runner_state.done
                rng_step = runner_state.rng

                rng_step, _rng_action = jax.random.split(rng_step)
                actions = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
                log_probs = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
                values = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
                actions_cont = jnp.zeros(
                    (num_envs, num_agents, adapter.max_action_dim), dtype=jnp.float32
                )
                new_actor_hs_all = []

                for team_idx, team in enumerate(team_specs):
                    idx = team_indices[team_idx]
                    n_team = len(team.agent_indices)
                    team_obs = obs_a[:, idx, : team.obs_dim]
                    team_positions = positions[:, idx, :]
                    team_actor_hs = actor_hs[team_idx]

                    all_msgs = _message_step(
                        actor_states[team_idx].apply_fn,
                        actor_states[team_idx].params,
                        team_actor_hs,
                        team_obs,
                        n_team,
                    )
                    if stop_neighbor_msg_grad:
                        all_msgs = jax.lax.stop_gradient(all_msgs)
                    neighbor_msgs, neighbor_mask = gather_neighbors_by_comm_range(
                        all_msgs,
                        team_positions,
                        comm_range,
                        max_neighbor_slots,
                    )
                    batch = n_team * num_envs
                    done_flat = jnp.broadcast_to(
                        last_done[jnp.newaxis, :], (n_team, num_envs)
                    ).reshape(batch)
                    new_actor_hs, pi, _ = _act_batched(
                        actor_states[team_idx].apply_fn,
                        actor_states[team_idx].params,
                        team_actor_hs.reshape(batch, -1),
                        all_msgs.reshape(batch, msg_dim),
                        neighbor_msgs.transpose(0, 2, 1, 3).reshape(
                            batch, max_neighbor_slots, msg_dim
                        ),
                        neighbor_mask.transpose(0, 2, 1).reshape(
                            batch, max_neighbor_slots
                        ),
                        done_flat,
                    )
                    new_actor_hs = new_actor_hs.reshape(n_team, num_envs, hidden_size)
                    logits = pi.logits.reshape(n_team, num_envs, team.action_dim)
                    logits = _mask_team_logits(logits, team, action_dims_py)

                    team_global = team_critic_state_from_unmasked(
                        global_state,
                        idx,
                        num_agents,
                        max_obs_dim,
                    )
                    team_values = _team_critic_values(
                        critic_states[team_idx].apply_fn,
                        critic_states[team_idx].params,
                        team_global,
                        n_team,
                    )

                    _rng_action, key_team = jax.random.split(_rng_action)
                    action_keys = jax.random.split(key_team, n_team * num_envs).reshape(
                        n_team, num_envs, -1
                    )
                    team_action = jax.vmap(
                        lambda keys_row, lg_row: jax.vmap(
                            lambda k, l: distrax.Categorical(l).sample(seed=k)
                        )(keys_row, lg_row)
                    )(action_keys, logits)
                    team_log_prob = jax.vmap(
                        lambda lg_row, act_row: distrax.Categorical(lg_row).log_prob(
                            act_row
                        )
                    )(logits, team_action)

                    actions = actions.at[:, idx].set(team_action.transpose(1, 0))
                    log_probs = log_probs.at[:, idx].set(team_log_prob.transpose(1, 0))
                    values = values.at[:, idx].set(team_values.transpose(1, 0))
                    new_actor_hs_all.append(new_actor_hs)

                rng_step, _rng_env = jax.random.split(rng_step)
                step_keys = jax.random.split(_rng_env, num_envs)

                def step_one_env(key, state, act_d, act_c):
                    return adapter.step(key, state, act_d, act_c)[:6]

                (
                    new_obs,
                    new_unmasked,
                    new_positions,
                    new_env_state,
                    rewards,
                    new_done,
                ) = jax.vmap(step_one_env)(step_keys, env_state, actions, actions_cont)
                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)

                transition = Transition(
                    obs=obs_a,
                    global_state=global_state,
                    positions=positions,
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
                        env_state=new_env_state,
                        obs=new_obs,
                        unmasked_obs=new_unmasked,
                        positions=new_positions,
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

            global_state_last = runner_state.unmasked_obs
            last_values_all = []
            for team_idx, team in enumerate(team_specs):
                team_global = team_critic_state_from_unmasked(
                    global_state_last,
                    team_indices[team_idx],
                    num_agents,
                    max_obs_dim,
                )
                last_v = _team_critic_values(
                    runner_state.critic_train_states[team_idx].apply_fn,
                    runner_state.critic_train_states[team_idx].params,
                    team_global,
                    len(team.agent_indices),
                )
                last_values_all.append(last_v)

            rng_update, *team_rngs = jax.random.split(
                runner_state.rng, len(team_specs) + 1
            )
            new_actor_states = []
            new_critic_states = []
            loss_infos = []
            for team_idx, team in enumerate(team_specs):
                team_traj = _slice_team_traj(
                    traj_batch,
                    team_indices[team_idx],
                    num_agents,
                    max_obs_dim,
                )
                new_a, new_c, loss_i = _ppo_update_team(
                    team,
                    runner_state.actor_train_states[team_idx],
                    runner_state.critic_train_states[team_idx],
                    init_actor_hs[team_idx],
                    team_traj,
                    last_values_all[team_idx],
                    team_rngs[team_idx],
                )
                new_actor_states.append(new_a)
                new_critic_states.append(new_c)
                loss_infos.append(loss_i)

            actor_train_states = tuple(new_actor_states)
            critic_train_states = tuple(new_critic_states)
            loss_info = jax.tree.map(lambda *xs: jnp.stack(xs), *loss_infos)
            loss_info = jax.tree.map(lambda x: x.mean(axis=(0, 1, 2)), loss_info)

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
            returns_avg = jnp.where(
                num_episodes > 0, ret_at_done.sum() / num_episodes, 0.0
            )
            episode_length_avg = jnp.where(
                num_episodes > 0, len_at_done.sum() / num_episodes, 0.0
            )

            metric = {
                "update_step": runner_state.update_step,
                "env_step": (
                    runner_state.update_step
                    * config["num_envs"]
                    * config["num_steps_per_env_per_update"]
                ),
                "return": returns_avg,
                "episode_length": episode_length_avg,
                "num_episodes_completed": num_episodes,
                "mean_reward_per_step": reward.mean(),
                "actor_loss": loss_info["actor_loss"].mean(),
                "value_loss": loss_info["value_loss"].mean(),
                "entropy": loss_info["entropy"].mean(),
                "approx_kl": loss_info["approx_kl"].mean(),
                "total_loss": loss_info["total_loss"].mean(),
                "grad_norm_actor": loss_info["grad_norm_actor"].mean(),
                "grad_norm_critic": loss_info["grad_norm_critic"].mean(),
                "state_mse": loss_info["state_mse"].mean(),
            }
            for team_idx, team in enumerate(team_specs):
                team_r = reward[:, :, team_indices[team_idx]].mean(axis=-1)
                (_, _), (team_ret_done, team_len_done) = jax.lax.scan(
                    _rollout_episode_metrics,
                    (
                        jnp.zeros_like(runner_state.cumulative_return),
                        jnp.zeros_like(initial_timesteps),
                    ),
                    (team_r, done),
                )
                metric[f"return_team_{team.name}"] = jnp.where(
                    num_episodes > 0, team_ret_done.sum() / num_episodes, 0.0
                )
                metric[f"episode_length_team_{team.name}"] = jnp.where(
                    num_episodes > 0, team_len_done.sum() / num_episodes, 0.0
                )

            def callback(exp_id, metric_dict):
                if LOGGER is not None:
                    log_step = int(np.array(metric_dict["update_step"]))
                    LOGGER.log(
                        int(exp_id),
                        {k: np.array(v) for k, v in metric_dict.items()},
                        step=log_step,
                    )

            jax.experimental.io_callback(callback, None, exp_id, metric)

            if log_rollout_videos:
                log_now = jnp.logical_and(
                    exp_id == 0,
                    jnp.any(runner_state.update_step == checkpoint_steps),
                )
                actor_params = tuple(ts.params for ts in actor_train_states)
                jax.experimental.io_callback(
                    rollout_io_callback,
                    None,
                    log_now,
                    exp_id,
                    runner_state.update_step,
                    actor_params,
                )

            return (
                RunnerState(
                    actor_train_states=actor_train_states,
                    critic_train_states=critic_train_states,
                    actor_hidden_states=runner_state.actor_hidden_states,
                    env_state=runner_state.env_state,
                    obs=runner_state.obs,
                    unmasked_obs=runner_state.unmasked_obs,
                    positions=runner_state.positions,
                    done=runner_state.done,
                    cumulative_return=new_cumulative_return,
                    timesteps=runner_state.timesteps,
                    update_step=runner_state.update_step + 1,
                    rng=rng_update,
                ),
                metric,
            )

        rng, _train_rng = jax.random.split(rng)
        initial_runner = RunnerState(
            actor_train_states=actor_train_states,
            critic_train_states=critic_train_states,
            actor_hidden_states=actor_hidden_states,
            env_state=env_state,
            obs=obs,
            unmasked_obs=unmasked_obs,
            positions=positions,
            done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            cumulative_return=jnp.zeros((num_envs,), dtype=jnp.float32),
            timesteps=jnp.zeros((num_envs,), dtype=jnp.int32),
            update_step=0,
            rng=_train_rng,
        )

        final_runner, _ = jax.lax.scan(
            _train_loop,
            initial_runner,
            None,
            length=config["num_update_steps"],
        )
        return final_runner

    return train


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config_mappo_att_grucomm_dub_state",
)
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
            config["num_update_steps"]
            * config["num_epochs"]
            * config["num_minibatches"]
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
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Compiling MAPPO GRU comm MPE...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
