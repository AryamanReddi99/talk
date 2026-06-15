"""Unified team-shared MAPPO for JaxMARL MPE environments."""

import datetime
from typing import Any, NamedTuple, Optional

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
    flatten_time_actors,
    num_team_actors,
    reshape_flat_minibatches,
    shuffle_flat_batch,
    slot_agent_ids,
)
from talk.environments.mpe.rollout_viz import log_rollout_video_callback
from talk.networks.mlp import (
    ActorContinuous,
    ActorDiscrete,
    CriticContinuous,
    CriticDiscrete,
)
from talk.utils.jax_utils import pytree_norm, jprint
from talk.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER: Optional[WandbMultiLogger] = None


class Transition(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    action_discrete: IntArray
    action_cont: FloatArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class TeamTraj(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    action_discrete: IntArray
    action_cont: FloatArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class RunnerState(NamedTuple):
    actor_train_states: tuple
    critic_train_states: tuple
    env_state: Any
    obs: FloatArray
    unmasked_obs: FloatArray
    done: BoolArray
    cumulative_return: FloatArray
    timesteps: IntArray
    update_step: int
    rng: PRNGKeyArray


class UpdateState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    traj_batch: TeamTraj
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def _slice_team_traj(
    traj: Transition,
    indices: IntArray,
    cont_dim: int,
    num_agents: int,
    max_obs_dim: int,
) -> TeamTraj:
    return TeamTraj(
        obs=traj.obs[:, :, indices, :],
        global_state=team_critic_state_from_unmasked(
            traj.global_state,
            indices,
            num_agents,
            max_obs_dim,
        ),
        action_discrete=traj.action_discrete[:, :, indices],
        action_cont=traj.action_cont[:, :, indices, :cont_dim],
        log_prob=traj.log_prob[:, :, indices],
        reward=traj.reward[:, :, indices],
        done=traj.done[:, :, indices],
        new_done=traj.new_done[:, :, indices],
        value=traj.value[:, :, indices],
    )


def _calculate_gae(
    traj: TeamTraj, last_values: FloatArray, gamma: float, gae_lambda: float
) -> tuple[FloatArray, FloatArray]:
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
        last_values, traj.value, traj.reward, traj.new_done
    )
    return advantages.transpose(1, 2, 0), targets.transpose(1, 2, 0)


def make_train(config: dict):
    adapter = build_mpe_env(
        env_name=config["env_name"],
        env_kwargs=config.get("env_kwargs", {}),
        sight_range=config["sight_range"],
        num_agents=config.get("num_agents"),
    )
    num_agents = adapter.num_agents
    max_obs_dim = adapter.max_obs_dim
    max_action_dim = adapter.max_action_dim
    num_envs = config["num_envs"]
    team_specs = [t for t in adapter.team_specs if t.trainable]
    team_indices = [jnp.array(t.agent_indices, dtype=jnp.int32) for t in team_specs]
    team_state_dim = critic_state_dim(num_agents, max_obs_dim)

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
            reset_keys = jax.random.split(_rng_reset, num_envs)
            obs, unmasked_obs, _, env_state = jax.vmap(adapter.reset)(reset_keys)

            def make_lr_schedule(base_lr: float):
                if config["anneal_lr"]:

                    def linear_schedule(count):
                        frac = 1.0 - (count // config["num_gradient_steps"])
                        return base_lr * frac

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

            actor_states = []
            critic_states = []
            rng_local = rng
            for team in team_specs:
                rng_local, rng_actor, rng_critic = jax.random.split(rng_local, 3)
                if team.action_kind == "discrete":
                    actor = ActorDiscrete(
                        action_dim=team.action_dim,
                        activation=config["activation"],
                        hidden_dim=config["fc_dim_size"],
                    )
                else:
                    actor = ActorContinuous(
                        action_dim=team.action_dim,
                        activation=config["activation"],
                        hidden_dim=config["fc_dim_size"],
                    )
                if team.action_kind == "discrete":
                    critic = CriticDiscrete(
                        activation=config["activation"],
                        hidden_dim=config["fc_dim_size"],
                    )
                else:
                    critic = CriticContinuous(
                        activation=config["activation"],
                        hidden_dim=config["fc_dim_size"],
                    )
                dummy_obs = jnp.zeros((team.obs_dim,), dtype=jnp.float32)
                dummy_global = jnp.zeros((team_state_dim,), dtype=jnp.float32)
                actor_states.append(
                    TrainState.create(
                        apply_fn=actor.apply,
                        params=actor.init(rng_actor, dummy_obs),
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
            return (
                obs,
                unmasked_obs,
                env_state,
                tuple(actor_states),
                tuple(critic_states),
                rng_local,
            )

        rng, _rng_setup = jax.random.split(rng)
        obs, unmasked_obs, env_state, actor_train_states, critic_train_states, rng = train_setup(
            _rng_setup
        )

        def _ppo_update_team(
            team: TeamSpec,
            actor_ts: TrainState,
            critic_ts: TrainState,
            traj: TeamTraj,
            last_values: FloatArray,
            rng_u: PRNGKeyArray,
        ):
            advantages, targets = _calculate_gae(
                traj, last_values, config["gamma"], config["gae_lambda"]
            )
            update_state = UpdateState(
                actor_train_state=actor_ts,
                critic_train_state=critic_ts,
                traj_batch=traj,
                advantages=advantages,
                targets=targets,
                rng=rng_u,
            )

            def _update_epoch(update_state: UpdateState, _):
                traj_b = update_state.traj_batch
                adv = update_state.advantages
                tgt = update_state.targets
                rng_e = update_state.rng
                a_ts = update_state.actor_train_state
                c_ts = update_state.critic_train_state

                n_team = len(team.agent_indices)
                n_actors = num_team_actors(n_team, num_envs)
                num_steps = traj_b.obs.shape[0]
                batch_size = num_steps * n_actors
                slot_agents = slot_agent_ids(n_team, num_envs)
                slot_agents_tiled = jnp.tile(slot_agents, num_steps)
                obs_flat = flatten_time_actors(
                    traj_b.obs[:, :, :, : team.obs_dim], n_team, num_envs
                )
                gs_flat = flatten_time_actors(traj_b.global_state, n_team, num_envs)
                log_prob_flat = flatten_time_actors(traj_b.log_prob, n_team, num_envs)
                value_flat = flatten_time_actors(traj_b.value, n_team, num_envs)
                adv_flat = flatten_time_actors(adv, n_team, num_envs)
                tgt_flat = flatten_time_actors(tgt, n_team, num_envs)
                if team.action_kind == "discrete":
                    act_flat = flatten_time_actors(traj_b.action_discrete, n_team, num_envs)
                else:
                    act_flat = flatten_time_actors(
                        traj_b.action_cont[:, :, :, : team.action_dim],
                        n_team,
                        num_envs,
                    )
                rng_e, shuffled = shuffle_flat_batch(
                    rng_e,
                    batch_size,
                    (
                        obs_flat,
                        gs_flat,
                        log_prob_flat,
                        value_flat,
                        adv_flat,
                        tgt_flat,
                        act_flat,
                        slot_agents_tiled,
                    ),
                )
                num_mb = config["num_minibatches"]
                minibatches = tuple(reshape_flat_minibatches(x, num_mb) for x in shuffled)

                def _update_minibatch(carry, minibatch):
                    a_ts_i, c_ts_i = carry
                    (
                        obs_mb,
                        gs_mb,
                        log_prob_mb,
                        value_mb,
                        adv_mb,
                        tgt_mb,
                        act_mb,
                        slot_agents_mb,
                    ) = minibatch

                    def _actor_loss(params, obs_mb, act_mb, log_prob_mb, adv_mb, slot_agents_mb):
                        old_log_prob = log_prob_mb
                        adv_flat = adv_mb
                        if team.action_kind == "discrete":
                            pi = a_ts_i.apply_fn(params, obs_mb)
                            agent_dims = jnp.array(
                                [adapter.action_dims_py[i] for i in team.agent_indices],
                                dtype=jnp.int32,
                            )
                            dims = agent_dims[slot_agents_mb]
                            valid = jnp.arange(team.action_dim)[None, :] < dims[:, None]
                            masked_logits = jnp.where(valid, pi.logits, -1e10)
                            masked_pi = distrax.Categorical(logits=masked_logits)
                            log_prob = masked_pi.log_prob(act_mb)
                            entropy = masked_pi.entropy().mean()
                        else:
                            pi = a_ts_i.apply_fn(params, obs_mb)
                            log_prob = pi.log_prob(act_mb)
                            entropy = pi.entropy().mean()
                        logratio = log_prob - old_log_prob
                        ratio = jnp.exp(logratio)
                        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                        loss_1 = ratio * adv_flat
                        loss_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * adv_flat
                        )
                        loss_actor = -jnp.minimum(loss_1, loss_2).mean()
                        loss = loss_actor - config["ent_coef"] * entropy
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio - 1) - logratio).mean(),
                        }

                    def _critic_loss(params, gs_mb, value_mb, tgt_mb):
                        gs = gs_mb.reshape(-1, team_state_dim)
                        value = c_ts_i.apply_fn(params, gs).reshape(value_mb.shape)
                        value_clipped = value_mb + (value - value_mb).clip(
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
                    )(
                        a_ts_i.params,
                        obs_mb,
                        act_mb,
                        log_prob_mb,
                        adv_mb,
                        slot_agents_mb,
                    )
                    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                        _critic_loss, has_aux=True
                    )(c_ts_i.params, gs_mb, value_mb, tgt_mb)

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
                    UpdateState(
                        actor_train_state=a_ts,
                        critic_train_state=c_ts,
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

            def _env_step(runner_state: RunnerState, _):
                actor_states = runner_state.actor_train_states
                critic_states = runner_state.critic_train_states
                obs_a = runner_state.obs
                env_state = runner_state.env_state
                done = runner_state.done
                rng_step = runner_state.rng
                global_state = runner_state.unmasked_obs

                rng_step, _rng_action = jax.random.split(rng_step)
                actions_discrete = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
                actions_cont = jnp.zeros((num_envs, num_agents, max_action_dim), dtype=jnp.float32)
                log_probs = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
                values = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)

                for team_idx, team in enumerate(team_specs):
                    idx = team_indices[team_idx]
                    n_team = len(team.agent_indices)
                    team_obs = obs_a[:, idx, : team.obs_dim].reshape(-1, team.obs_dim)
                    team_global = team_critic_state_from_unmasked(
                        global_state,
                        idx,
                        num_agents,
                        max_obs_dim,
                    ).reshape(-1, team_state_dim)
                    _rng_action, key_team = jax.random.split(_rng_action)
                    pi = actor_states[team_idx].apply_fn(actor_states[team_idx].params, team_obs)
                    team_values = critic_states[team_idx].apply_fn(
                        critic_states[team_idx].params, team_global
                    )
                    if team.action_kind == "discrete":
                        logits = pi.logits
                        dims = jnp.array(
                            [adapter.action_dims_py[i] for i in team.agent_indices],
                            dtype=jnp.int32,
                        )
                        tiled_dims = jnp.repeat(dims[None, :], num_envs, axis=0).reshape(-1)
                        valid = jnp.arange(team.action_dim)[None, :] < tiled_dims[:, None]
                        masked_logits = jnp.where(valid, logits, -1e10)
                        masked_pi = distrax.Categorical(logits=masked_logits)
                        team_action = masked_pi.sample(seed=key_team)
                        team_log_prob = masked_pi.log_prob(team_action)
                        actions_discrete = actions_discrete.at[:, idx].set(
                            team_action.reshape(num_envs, n_team)
                        )
                    else:
                        team_action = pi.sample(seed=key_team)
                        team_log_prob = pi.log_prob(team_action)
                        actions_cont = actions_cont.at[:, idx, : team.action_dim].set(
                            team_action.reshape(num_envs, n_team, team.action_dim)
                        )
                    log_probs = log_probs.at[:, idx].set(team_log_prob.reshape(num_envs, n_team))
                    values = values.at[:, idx].set(team_values.reshape(num_envs, n_team))

                rng_step, _rng_env = jax.random.split(rng_step)
                step_keys = jax.random.split(_rng_env, num_envs)

                def step_one_env(key, state, act_d, act_c):
                    return adapter.step(key, state, act_d, act_c)[:6]

                new_obs, new_unmasked, _, new_env_state, rewards, new_done = jax.vmap(step_one_env)(
                    step_keys, env_state, actions_discrete, actions_cont
                )
                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)

                transition = Transition(
                    obs=obs_a,
                    global_state=global_state,
                    action_discrete=actions_discrete,
                    action_cont=actions_cont,
                    log_prob=log_probs,
                    reward=rewards,
                    done=jnp.broadcast_to(done[:, None], (num_envs, num_agents)),
                    new_done=jnp.broadcast_to(new_done[:, None], (num_envs, num_agents)),
                    value=values,
                )

                return (
                    RunnerState(
                        actor_train_states=actor_states,
                        critic_train_states=critic_states,
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

            global_state_last = runner_state.unmasked_obs
            last_values_all = []
            for team_idx, team in enumerate(team_specs):
                idx = team_indices[team_idx]
                n_team = len(team.agent_indices)
                team_global = team_critic_state_from_unmasked(
                    global_state_last,
                    idx,
                    num_agents,
                    max_obs_dim,
                )
                gs = team_global.reshape(-1, team_global.shape[-1])
                v = runner_state.critic_train_states[team_idx].apply_fn(
                    runner_state.critic_train_states[team_idx].params,
                    gs,
                )
                last_values_all.append(v.reshape(n_team, num_envs))

            rng_update, *team_rngs = jax.random.split(runner_state.rng, len(team_specs) + 1)
            new_actor_states = []
            new_critic_states = []
            loss_infos = []
            for team_idx, team in enumerate(team_specs):
                team_traj = _slice_team_traj(
                    traj_batch,
                    team_indices[team_idx],
                    team.action_dim,
                    num_agents,
                    max_obs_dim,
                )
                new_a, new_c, loss_i = _ppo_update_team(
                    team,
                    runner_state.actor_train_states[team_idx],
                    runner_state.critic_train_states[team_idx],
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
            done = traj_batch.new_done[:, :, 0]
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
            returns_avg = jnp.where(num_episodes > 0, ret_at_done.sum() / num_episodes, 0.0)
            episode_length_avg = jnp.where(num_episodes > 0, len_at_done.sum() / num_episodes, 0.0)

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
        initial_runner = RunnerState(
            actor_train_states=actor_train_states,
            critic_train_states=critic_train_states,
            env_state=env_state,
            obs=obs,
            unmasked_obs=unmasked_obs,
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


@hydra.main(version_base=None, config_path=".", config_name="config_mappo")
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
            job_type=config["algorithm"] + config["custom_name"],
            group=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            config=config,
            mode="online" if config["use_wandb"] else "disabled",
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Compiling MAPPO MPE...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
