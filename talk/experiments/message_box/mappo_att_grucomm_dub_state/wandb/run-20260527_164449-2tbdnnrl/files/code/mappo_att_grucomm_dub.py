"""MAPPO with shared pre-attention GRU actors/critics on MessageBox."""

import datetime
from typing import Any, NamedTuple, Tuple

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from talk.environments.message_box import MessageBox
from talk.networks.gru import ActorDiscretePreAttnCommRNN, CriticDiscreteRNN, ScannedRNN
from talk.utils.jax_utils import pytree_norm
from talk.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None

MAX_NEIGHBOR_SLOTS = 2


class Transition(NamedTuple):
    obs: FloatArray
    global_state: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray


class RunnerState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    actor_hidden_states: FloatArray
    critic_hidden_states: FloatArray
    env_state: Any
    obs: FloatArray
    done: BoolArray
    cumulative_return: FloatArray
    timesteps: IntArray
    update_step: int
    rng: PRNGKeyArray


class UpdateState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    init_actor_h: FloatArray
    init_critic_h: FloatArray
    traj_batch: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def _global_state(obs: FloatArray) -> FloatArray:
    return obs.reshape(obs.shape[:-2] + (-1,))


def _done_for_rnn(done: BoolArray) -> BoolArray:
    if done.ndim == 1:
        return done[jnp.newaxis, :]
    return done


def _gather_neighbor_slots(
    all_msgs: FloatArray, num_agents: int, agent_axis: int = 0
) -> Tuple[FloatArray, FloatArray]:
    """Fixed [left, right] neighbor slots with mask."""
    moved = jnp.moveaxis(all_msgs, agent_axis, 0)
    left = jnp.concatenate([jnp.zeros_like(moved[:1]), moved[:-1]], axis=0)
    right = jnp.concatenate([moved[1:], jnp.zeros_like(moved[:1])], axis=0)
    neighbor_msgs = jnp.stack([left, right], axis=1)
    agent_idx = jnp.arange(num_agents)
    left_mask = (agent_idx > 0).astype(jnp.float32)
    right_mask = (agent_idx < num_agents - 1).astype(jnp.float32)
    neighbor_mask = jnp.stack([left_mask, right_mask], axis=-1)
    spatial_shape = moved.shape[1:-1]
    neighbor_mask = jnp.broadcast_to(
        neighbor_mask.reshape(
            num_agents, MAX_NEIGHBOR_SLOTS, *([1] * len(spatial_shape))
        ),
        (num_agents, MAX_NEIGHBOR_SLOTS, *spatial_shape),
    )
    return neighbor_msgs, neighbor_mask


def _reshape_rnn_minibatches(x: FloatArray, num_minibatches: int) -> FloatArray:
    """(T, E, ...) -> (num_minibatches, T, mb_E, ...)."""
    return jnp.swapaxes(
        jnp.reshape(x, (x.shape[0], num_minibatches, -1, *x.shape[2:])),
        1,
        0,
    )


def _message_step(
    actor_apply,
    params,
    actor_hs: FloatArray,
    obs: FloatArray,
    num_agents: int,
) -> FloatArray:
    """Compute broadcast messages from concat(obs, hidden). obs: (E, A, D)."""
    num_envs = obs.shape[0]
    obs_ae = obs.transpose(1, 0, 2)
    batch = num_agents * num_envs
    msgs = actor_apply(
        params,
        actor_hs.reshape(batch, -1),
        obs_ae.reshape(batch, obs_ae.shape[-1]),
        method=ActorDiscretePreAttnCommRNN.message,
    )
    msg_dim = msgs.shape[-1]
    return msgs.reshape(num_agents, num_envs, msg_dim)


def _actor_trajectory(
    actor_apply,
    params,
    init_hs: FloatArray,
    obs: FloatArray,
    done: BoolArray,
    num_agents: int,
) -> FloatArray:
    """Run message -> attention -> GRU -> action over time. Returns logits (T, E, A, A_dim)."""

    def step(h, inputs):
        obs_t, done_t = inputs
        msgs = _message_step(actor_apply, params, h, obs_t, num_agents)
        neighbor_msgs, neighbor_mask = _gather_neighbor_slots(
            msgs, num_agents, agent_axis=0
        )
        batch = num_agents * obs_t.shape[0]
        done_flat = jnp.broadcast_to(
            done_t[jnp.newaxis, :], (num_agents, obs_t.shape[0])
        ).reshape(batch)
        new_h, pi = _act_batched(
            actor_apply,
            params,
            h.reshape(batch, -1),
            msgs.reshape(batch, -1),
            neighbor_msgs.transpose(0, 2, 1, 3).reshape(
                batch, MAX_NEIGHBOR_SLOTS, msgs.shape[-1]
            ),
            neighbor_mask.transpose(0, 2, 1).reshape(batch, MAX_NEIGHBOR_SLOTS),
            done_flat,
        )
        logits = pi.logits.reshape(num_agents, obs_t.shape[0], -1)
        return new_h.reshape(num_agents, obs_t.shape[0], -1), logits

    _, logits = jax.lax.scan(step, init_hs, (obs, done))
    return logits.transpose(0, 2, 1, 3)


def _act_batched(
    actor_apply,
    params,
    hidden: FloatArray,
    own_msgs: FloatArray,
    neighbor_msgs: FloatArray,
    neighbor_mask: FloatArray,
    dones: FloatArray,
) -> Tuple[FloatArray, distrax.Distribution]:
    """hidden/own_msgs: (N, D); neighbor_msgs: (N, S, D); dones: (N,)."""
    return actor_apply(
        params,
        hidden,
        own_msgs,
        neighbor_msgs,
        neighbor_mask,
        dones,
        method=ActorDiscretePreAttnCommRNN.act,
    )


def _critic_step(
    critic_apply,
    params,
    critic_hs: FloatArray,
    global_state: FloatArray,
    done: BoolArray,
    num_agents: int,
) -> Tuple[FloatArray, FloatArray]:
    num_envs = global_state.shape[0]
    batch = num_agents * num_envs
    gs = jnp.broadcast_to(
        global_state[jnp.newaxis, :, :],
        (num_agents, num_envs, global_state.shape[-1]),
    )
    cr_in = (
        gs.reshape(1, batch, -1),
        jnp.broadcast_to(done[jnp.newaxis, :], (num_agents, num_envs)).reshape(1, batch),
    )
    new_hs, values = critic_apply(
        params,
        critic_hs.reshape(batch, -1),
        cr_in,
    )
    return new_hs.reshape(num_agents, num_envs, -1), values.reshape(num_agents, num_envs)


def _calculate_gae_batched(
    traj: Transition,
    last_values: FloatArray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[FloatArray, FloatArray]:
    """last_values: (A, E); traj fields on (T, E, A)."""
    values = traj.value
    rewards = traj.reward
    dones = jnp.broadcast_to(traj.new_done[:, :, None], values.shape)

    def gae_one_agent(last_v, val, rew, done):
        def step(carry, transition):
            gae, next_value = carry
            delta = (
                transition[0]
                + gamma * next_value * (1 - transition[1])
                - transition[2]
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


def make_train(config: dict):
    env = MessageBox(
        num_agents=config["num_agents"],
        max_steps=config["max_steps"],
        reward_correct=config["reward_correct"],
        reward_wrong=config["reward_wrong"],
        include_agent_id=config.get("include_agent_id", False),
    )
    num_agents = env.num_agents
    max_action_dim = max(env.action_space(i).n for i in range(num_agents))
    obs_dim = env.obs_dim
    global_state_dim = num_agents * obs_dim
    msg_dim = config["fc_dim_size"]
    hidden_size = config["gru_hidden_size"]
    num_envs = config["num_envs"]
    actor_action_mask = jnp.array(
        [env.action_space(i).n > 1 for i in range(num_agents)], dtype=jnp.float32
    )

    config["batch_shuffle_dim"] = num_envs

    def train(rng: PRNGKeyArray, exp_id: int):
        def train_setup(rng: PRNGKeyArray):
            rng, _rng_reset = jax.random.split(rng)
            reset_keys = jax.random.split(_rng_reset, num_envs)
            obs, env_state = jax.vmap(env.reset)(reset_keys)

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

            rng, rng_actor, rng_critic = jax.random.split(rng, 3)
            actor = ActorDiscretePreAttnCommRNN(
                action_dim=max_action_dim,
                fc_dim_size=config["fc_dim_size"],
                msg_dim=msg_dim,
                activation=config["activation"],
            )
            critic = CriticDiscreteRNN(
                hidden_size=hidden_size,
                fc_dim_size=config["fc_dim_size"],
                activation=config["activation"],
            )

            ac_init_h = ScannedRNN.initialize_carry(num_agents * num_envs, hidden_size)
            cr_init_h = ScannedRNN.initialize_carry(num_agents * num_envs, hidden_size)
            ac_in = (
                jnp.zeros((1, num_agents * num_envs, obs_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents * num_envs), dtype=bool),
                jnp.zeros(
                    (num_agents * num_envs, MAX_NEIGHBOR_SLOTS, msg_dim),
                    dtype=jnp.float32,
                ),
                jnp.zeros((num_agents * num_envs, MAX_NEIGHBOR_SLOTS), dtype=jnp.float32),
            )
            cr_in = (
                jnp.zeros((1, num_agents * num_envs, global_state_dim), dtype=jnp.float32),
                jnp.zeros((1, num_agents * num_envs), dtype=bool),
            )

            actor_ts = TrainState.create(
                apply_fn=actor.apply,
                params=actor.init(rng_actor, ac_init_h, ac_in),
                tx=tx_actor,
            )
            critic_ts = TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(rng_critic, cr_init_h, cr_in),
                tx=tx_critic,
            )

            actor_hs = ac_init_h.reshape(num_agents, num_envs, hidden_size)
            critic_hs = cr_init_h.reshape(num_agents, num_envs, hidden_size)
            return obs, env_state, actor_ts, critic_ts, actor_hs, critic_hs

        rng, _rng_setup = jax.random.split(rng)
        obs, env_state, actor_train_state, critic_train_state, actor_hidden_states, critic_hidden_states = (
            train_setup(_rng_setup)
        )

        def _ppo_update(
            actor_ts: TrainState,
            critic_ts: TrainState,
            init_actor_hs: FloatArray,
            init_critic_hs: FloatArray,
            traj: Transition,
            last_values: FloatArray,
            rng: PRNGKeyArray,
        ):
            advantages, targets = _calculate_gae_batched(
                traj, last_values, config["gamma"], config["gae_lambda"]
            )
            update_state = UpdateState(
                actor_train_state=actor_ts,
                critic_train_state=critic_ts,
                init_actor_h=init_actor_hs,
                init_critic_h=init_critic_hs,
                traj_batch=traj,
                advantages=advantages,
                targets=targets,
                rng=rng,
            )

            def _update_epoch(update_state: UpdateState, _):
                traj_b = update_state.traj_batch
                adv = update_state.advantages
                tgt = update_state.targets
                rng_e = update_state.rng
                actor_ts_e = update_state.actor_train_state
                critic_ts_e = update_state.critic_train_state

                rng_e, _rng_perm = jax.random.split(rng_e)
                permutation = jax.random.permutation(_rng_perm, num_envs)
                shuffled = (
                    jnp.take(init_actor_hs, permutation, axis=1),
                    jnp.take(init_critic_hs, permutation, axis=1),
                    jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), traj_b),
                    jnp.take(adv, permutation, axis=1),
                    jnp.take(tgt, permutation, axis=1),
                )
                traj_shuf = shuffled[2]
                num_mb = config["num_minibatches"]
                mb_actor_h = shuffled[0].reshape(num_agents, num_mb, -1, hidden_size).swapaxes(
                    0, 1
                )
                mb_critic_h = shuffled[1].reshape(num_agents, num_mb, -1, hidden_size).swapaxes(
                    0, 1
                )
                minibatches = (
                    mb_actor_h,
                    mb_critic_h,
                    jax.tree.map(
                        lambda x: _reshape_rnn_minibatches(x, num_mb), traj_shuf
                    ),
                    _reshape_rnn_minibatches(shuffled[3], num_mb),
                    _reshape_rnn_minibatches(shuffled[4], num_mb),
                )

                def _update_minibatch(carry, minibatch):
                    a_ts, c_ts = carry
                    h_a, h_c, traj_mb, adv_mb, tgt_mb = minibatch

                    def _actor_loss(params, init_h, traj_mb, adv_mb):
                        logits = _actor_trajectory(
                            a_ts.apply_fn,
                            params,
                            init_h,
                            traj_mb.obs,
                            traj_mb.done,
                            num_agents,
                        )
                        log_prob = distrax.Categorical(logits=logits).log_prob(
                            traj_mb.action
                        )
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
                        per_agent = -jnp.minimum(loss_1, loss_2)
                        mask = actor_action_mask[jnp.newaxis, jnp.newaxis, :]
                        loss_actor = (per_agent * mask).sum() / jnp.maximum(mask.sum(), 1.0)
                        entropy = (
                            distrax.Categorical(logits=logits).entropy() * mask
                        ).sum()
                        entropy = entropy / jnp.maximum(mask.sum(), 1.0)
                        logratio_flat = logratio.reshape(-1)
                        ratio_flat = ratio.reshape(-1)
                        loss = loss_actor - config["ent_coef"] * entropy
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio_flat - 1) - logratio_flat).mean(),
                        }

                    def _critic_loss(params, init_h, traj_mb, tgt_mb):
                        def critic_scan(carry, inputs):
                            hs, gs, done_t = carry, inputs[0], inputs[1]
                            new_hs, values = _critic_step(
                                c_ts.apply_fn,
                                params,
                                hs,
                                gs,
                                done_t,
                                num_agents,
                            )
                            return new_hs, values

                        _, values = jax.lax.scan(
                            critic_scan,
                            init_h,
                            (traj_mb.global_state, traj_mb.done),
                        )
                        value_clipped = traj_mb.value + (
                            values.transpose(0, 2, 1) - traj_mb.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        per_agent = jnp.maximum(
                            jnp.square(values.transpose(0, 2, 1) - tgt_mb),
                            jnp.square(value_clipped - tgt_mb),
                        )
                        loss = 0.5 * per_agent.mean()
                        return loss, {"value_loss": loss}

                    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                        _actor_loss, has_aux=True
                    )(a_ts.params, h_a, traj_mb, adv_mb)
                    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                        _critic_loss, has_aux=True
                    )(c_ts.params, h_c, traj_mb, tgt_mb)

                    a_ts = a_ts.apply_gradients(grads=actor_grads)
                    c_ts = c_ts.apply_gradients(grads=critic_grads)
                    aux = {
                        **actor_aux,
                        **critic_aux,
                        "total_loss": actor_loss + critic_loss,
                        "grad_norm_actor": pytree_norm(actor_grads),
                        "grad_norm_critic": pytree_norm(critic_grads),
                    }
                    return (a_ts, c_ts), aux

                (a_ts, c_ts), loss_info = jax.lax.scan(
                    _update_minibatch,
                    (actor_ts_e, critic_ts_e),
                    minibatches,
                )
                return (
                    UpdateState(
                        actor_train_state=a_ts,
                        critic_train_state=c_ts,
                        init_actor_h=init_actor_hs,
                        init_critic_h=init_critic_hs,
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
            return update_state.actor_train_state, update_state.critic_train_state, loss_info

        def _train_loop(runner_state: RunnerState, _):
            initial_timesteps = runner_state.timesteps
            init_actor_hs = runner_state.actor_hidden_states
            init_critic_hs = runner_state.critic_hidden_states

            def _env_step(runner_state: RunnerState, _):
                actor_ts = runner_state.actor_train_state
                critic_ts = runner_state.critic_train_state
                actor_hs = runner_state.actor_hidden_states
                critic_hs = runner_state.critic_hidden_states
                obs_a = runner_state.obs
                env_state = runner_state.env_state
                last_done = runner_state.done
                rng = runner_state.rng

                def reset_if_done(obs, state, done_flag, key):
                    return jax.lax.cond(done_flag, lambda: env.reset(key), lambda: (obs, state))

                rng, _rng_reset = jax.random.split(rng)
                reset_keys = jax.random.split(_rng_reset, num_envs)
                obs_a, env_state = jax.vmap(reset_if_done)(
                    obs_a, env_state, last_done, reset_keys
                )
                global_state = _global_state(obs_a)

                rng, _rng_action = jax.random.split(rng)
                action_keys = jax.random.split(
                    _rng_action, num_agents * num_envs
                ).reshape(num_agents, num_envs, -1)

                all_msgs = _message_step(
                    actor_ts.apply_fn,
                    actor_ts.params,
                    actor_hs,
                    obs_a,
                    num_agents,
                )
                neighbor_msgs, neighbor_mask = _gather_neighbor_slots(
                    all_msgs, num_agents, agent_axis=0
                )

                batch = num_agents * num_envs
                done_flat = jnp.broadcast_to(
                    last_done[jnp.newaxis, :], (num_agents, num_envs)
                ).reshape(batch)
                new_actor_hs, pi = _act_batched(
                    actor_ts.apply_fn,
                    actor_ts.params,
                    actor_hs.reshape(batch, -1),
                    all_msgs.reshape(batch, msg_dim),
                    neighbor_msgs.transpose(0, 2, 1, 3).reshape(
                        batch, MAX_NEIGHBOR_SLOTS, msg_dim
                    ),
                    neighbor_mask.transpose(0, 2, 1).reshape(batch, MAX_NEIGHBOR_SLOTS),
                    done_flat,
                )
                new_actor_hs = new_actor_hs.reshape(num_agents, num_envs, hidden_size)
                logits = pi.logits.reshape(num_agents, num_envs, max_action_dim)

                new_critic_hs, values = _critic_step(
                    critic_ts.apply_fn,
                    critic_ts.params,
                    critic_hs,
                    global_state,
                    last_done,
                    num_agents,
                )

                actions = jax.vmap(
                    lambda keys, lg: jax.vmap(
                        lambda k, l: distrax.Categorical(l).sample(seed=k)
                    )(keys, lg)
                )(action_keys, logits)
                log_probs = jax.vmap(
                    lambda lg, act: distrax.Categorical(lg).log_prob(act)
                )(logits, actions)

                rng, _rng_step = jax.random.split(rng)
                step_keys = jax.random.split(_rng_step, num_envs)

                def step_one_env(key, state, actions_per_agent):
                    return env.step(key, state, actions_per_agent)[:4]

                new_obs, new_env_state, rewards, new_done = jax.vmap(
                    step_one_env, in_axes=(0, 0, 1)
                )(step_keys, env_state, actions)

                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)

                transition = Transition(
                    obs=obs_a,
                    global_state=global_state,
                    action=actions.transpose(1, 0),
                    log_prob=log_probs.transpose(1, 0),
                    reward=rewards,
                    done=last_done,
                    new_done=new_done,
                    value=values.transpose(1, 0),
                )

                return (
                    RunnerState(
                        actor_train_state=actor_ts,
                        critic_train_state=critic_ts,
                        actor_hidden_states=new_actor_hs,
                        critic_hidden_states=new_critic_hs,
                        env_state=new_env_state,
                        obs=new_obs,
                        done=new_done,
                        cumulative_return=runner_state.cumulative_return,
                        timesteps=timesteps,
                        update_step=runner_state.update_step,
                        rng=rng,
                    ),
                    transition,
                )

            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["num_steps_per_env_per_update"],
            )

            global_state_last = _global_state(runner_state.obs)
            _, last_values = _critic_step(
                runner_state.critic_train_state.apply_fn,
                runner_state.critic_train_state.params,
                runner_state.critic_hidden_states,
                global_state_last,
                runner_state.done,
                num_agents,
            )

            rng, _rng_update = jax.random.split(runner_state.rng)
            actor_train_state, critic_train_state, loss_info = _ppo_update(
                runner_state.actor_train_state,
                runner_state.critic_train_state,
                init_actor_hs,
                init_critic_hs,
                traj_batch,
                last_values,
                _rng_update,
            )
            loss_info = jax.tree.map(lambda x: x.mean(axis=(0, 1)), loss_info)

            reward = traj_batch.reward
            done = traj_batch.new_done
            reward_agent0 = reward[:, :, 0]

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
                (reward_agent0, done),
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
                    * num_envs
                    * config["num_steps_per_env_per_update"]
                ),
                "return": returns_avg,
                "episode_length": episode_length_avg,
                "num_episodes_completed": num_episodes,
                "mean_reward_per_step": reward.mean(),
                "actor_loss": loss_info["actor_loss"],
                "value_loss": loss_info["value_loss"],
                "entropy": loss_info["entropy"],
                "approx_kl": loss_info["approx_kl"],
                "total_loss": loss_info["total_loss"],
            }
            for i in range(num_agents):
                metric[f"return_agent_{i}"] = returns_avg
                metric[f"mean_reward_agent_{i}"] = reward[:, :, i].mean()

            def callback(exp_id, metric_dict):
                if LOGGER is not None:
                    LOGGER.log(
                        int(exp_id), {k: np.array(v) for k, v in metric_dict.items()}
                    )

            jax.experimental.io_callback(callback, None, exp_id, metric)

            return (
                RunnerState(
                    actor_train_state=actor_train_state,
                    critic_train_state=critic_train_state,
                    actor_hidden_states=runner_state.actor_hidden_states,
                    critic_hidden_states=runner_state.critic_hidden_states,
                    env_state=runner_state.env_state,
                    obs=runner_state.obs,
                    done=runner_state.done,
                    cumulative_return=new_cumulative_return,
                    timesteps=runner_state.timesteps,
                    update_step=runner_state.update_step + 1,
                    rng=rng,
                ),
                metric,
            )

        rng, _train_rng = jax.random.split(rng)
        initial_runner = RunnerState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            actor_hidden_states=actor_hidden_states,
            critic_hidden_states=critic_hidden_states,
            env_state=env_state,
            obs=obs,
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
    config_name="config_mappo_att_grucomm_dub",
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

        print("Compiling MAPPO with shared pre-attention GRU...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
