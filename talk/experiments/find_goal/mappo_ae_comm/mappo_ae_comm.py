"""MAPPO + simplified AEComm on FindGoal.

All agents share one actor and one centralized critic (parameter sharing via
batched forward passes over num_agents * num_envs, not per-agent modules).
"""

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

from talk.environments.find_goal import FindGoal, NUM_ACTIONS
from talk.networks.ae_comm import AECommActor
from talk.networks.gru import CriticDiscreteRNN, ScannedRNN
from talk.utils.jax_utils import pytree_norm
from talk.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None


class Transition(NamedTuple):
    obs: FloatArray
    selfpos: FloatArray
    prev_msgs: FloatArray
    global_state: FloatArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray
    new_done: BoolArray
    value: FloatArray
    ae_loss: FloatArray


class RunnerState(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState
    actor_hidden_states: FloatArray
    critic_hidden_states: FloatArray
    prev_msgs: FloatArray
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


def _global_state(obs: FloatArray, selfpos: FloatArray) -> FloatArray:
    """Flatten all agents' POV and positions into one state vector per env."""
    pov_flat = obs.reshape(obs.shape[0], obs.shape[1], -1)
    return jnp.concatenate([pov_flat, selfpos.astype(jnp.float32)], axis=-1).reshape(
        obs.shape[0], -1
    )


def _reshape_rnn_minibatches(x: FloatArray, num_minibatches: int) -> FloatArray:
    return jnp.swapaxes(
        jnp.reshape(x, (x.shape[0], num_minibatches, -1, *x.shape[2:])),
        1,
        0,
    )


def _agent_env_batch(
    obs: FloatArray,
    selfpos: FloatArray,
    prev_msgs: FloatArray,
    num_agents: int,
    num_envs: int,
) -> Tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Flatten (agent, env) into one batch for shared-weight actor forward."""
    batch = num_agents * num_envs
    pov = obs.transpose(1, 0, 2, 3, 4).reshape(batch, *obs.shape[2:])
    sp = selfpos.transpose(1, 0, 2).reshape(batch, 2)
    prev_b = jnp.broadcast_to(
        prev_msgs[jnp.newaxis, :, :, :],
        (num_agents, num_envs, num_agents, prev_msgs.shape[-1]),
    ).reshape(batch, num_agents, prev_msgs.shape[-1])
    agent_idx = jnp.repeat(jnp.arange(num_agents), num_envs)
    return pov, sp, prev_b, agent_idx


def _actor_step_batched(
    apply_fn,
    params,
    actor_hs: FloatArray,
    obs: FloatArray,
    selfpos: FloatArray,
    prev_msgs: FloatArray,
    done: BoolArray,
    num_agents: int,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """One shared actor forward for all agents. obs: (E, A, ...)."""
    num_envs = obs.shape[0]
    batch = num_agents * num_envs
    pov, sp, prev_b, agent_idx = _agent_env_batch(
        obs, selfpos, prev_msgs, num_agents, num_envs
    )
    hidden = actor_hs.reshape(batch, -1)
    done_flat = jnp.broadcast_to(
        done[jnp.newaxis, :], (num_agents, num_envs)
    ).reshape(batch)

    feat = apply_fn(params, pov, sp, method=AECommActor.encode_obs)
    msg, ae_loss = apply_fn(params, feat, method=AECommActor.encode_message)
    new_h, pi, _ = apply_fn(
        params,
        hidden,
        pov,
        sp,
        prev_b,
        msg,
        agent_idx,
        done_flat,
        method=AECommActor.act,
    )
    hs = new_h.reshape(num_agents, num_envs, -1)
    logits = pi.logits.reshape(num_agents, num_envs, -1)
    msgs = msg.reshape(num_agents, num_envs, -1)
    ae_losses = ae_loss.reshape(num_agents, num_envs)
    return hs, logits, msgs, ae_losses


def _actor_rollout_step(
    apply_fn,
    params,
    actor_hs: FloatArray,
    obs: FloatArray,
    selfpos: FloatArray,
    prev_msgs: FloatArray,
    done: BoolArray,
    num_agents: int,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """One timestep: obs (E,A,...), returns hs (A,E,H), logits (A,E,A_dim), msgs (A,E,L), ae (A,E)."""
    return _actor_step_batched(
        apply_fn, params, actor_hs, obs, selfpos, prev_msgs, done, num_agents
    )


def _actor_trajectory(
    apply_fn,
    params,
    init_hs: FloatArray,
    obs: FloatArray,
    selfpos: FloatArray,
    prev_msgs: FloatArray,
    done: BoolArray,
    num_agents: int,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Scan over time. Returns logits (T,E,A,A_dim), ae_loss (T,E,A)."""

    def step(hs, inputs):
        obs_t, sp_t, pm_t, done_t = inputs
        new_hs, logits, _, ae_loss = _actor_rollout_step(
            apply_fn, params, hs, obs_t, sp_t, pm_t, done_t, num_agents
        )
        return new_hs, (logits.transpose(1, 0, 2), ae_loss.transpose(1, 0))

    _, (logits, ae_loss) = jax.lax.scan(
        step,
        init_hs,
        (obs, selfpos, prev_msgs, done),
    )
    return logits, ae_loss


def _critic_values(
    critic_apply,
    params,
    critic_hs: FloatArray,
    global_state: FloatArray,
    done: BoolArray,
) -> Tuple[FloatArray, FloatArray]:
    cr_in = (global_state[jnp.newaxis, ...], done[jnp.newaxis, ...])
    new_hs, values = critic_apply(params, critic_hs, cr_in)
    return new_hs, values.squeeze(0)


def _calculate_gae(
    traj: Transition,
    last_values: FloatArray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[FloatArray, FloatArray]:
    """Shared centralized value (E,); rewards/advantages per agent (T,E,A)."""
    values = traj.value
    rewards = traj.reward
    dones = jnp.broadcast_to(traj.new_done[:, :, None], rewards.shape)

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

    advantages, targets = jax.vmap(gae_one_agent, in_axes=(None, 2, 2, 2))(
        last_values, values, rewards, dones
    )
    return advantages.transpose(1, 2, 0), targets.transpose(1, 2, 0)


def _eval_test_episodes(
    env: FindGoal,
    apply_fn,
    params,
    rng: PRNGKeyArray,
    num_agents: int,
    hidden_size: int,
    comm_len: int,
    max_steps: int,
    num_test_episodes: int,
) -> Tuple[FloatArray, FloatArray]:
    """Run full greedy-policy episodes in parallel; return mean return and length."""

    def _run_one_episode(ep_rng: PRNGKeyArray) -> Tuple[FloatArray, FloatArray]:
        ep_rng, reset_key = jax.random.split(ep_rng)
        obs, state = env.reset(reset_key)
        obs_b = obs[jnp.newaxis, ...]
        actor_hs = ScannedRNN.initialize_carry(num_agents, hidden_size).reshape(
            num_agents, 1, hidden_size
        )
        prev_msgs = jnp.zeros((1, num_agents, comm_len), dtype=jnp.float32)
        last_done = jnp.zeros((1,), dtype=jnp.bool_)

        init_carry = (
            obs_b,
            state,
            actor_hs,
            prev_msgs,
            last_done,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False),
            ep_rng,
        )

        def eval_step(carry, _):
            (
                obs_b,
                state,
                actor_hs,
                prev_msgs,
                last_done,
                ep_return,
                ep_len,
                finished,
                step_rng,
            ) = carry

            def active_step(c):
                obs_b, state, actor_hs, prev_msgs, last_done, ep_return, ep_len, finished, step_rng = c
                selfpos_b = state.agent_pos[jnp.newaxis, ...]
                new_hs, logits, msgs, _ = _actor_rollout_step(
                    apply_fn,
                    params,
                    actor_hs,
                    obs_b,
                    selfpos_b,
                    prev_msgs,
                    last_done,
                    num_agents,
                )
                actions = jnp.argmax(logits, axis=-1)[:, 0]
                actions = jnp.where(state.agent_done, NUM_ACTIONS - 1, actions)
                step_rng, step_key = jax.random.split(step_rng)
                obs, new_state, rewards, done, _ = env.step_env(
                    step_key, state, actions
                )
                new_prev_msgs = msgs.transpose(1, 0, 2)
                return (
                    obs[jnp.newaxis, ...],
                    new_state,
                    new_hs,
                    new_prev_msgs,
                    done[jnp.newaxis],
                    ep_return + rewards.sum(),
                    ep_len + 1.0,
                    done,
                    step_rng,
                )

            return jax.lax.cond(finished, lambda c: c, active_step, carry), None

        final_carry, _ = jax.lax.scan(
            eval_step, init_carry, None, length=max_steps
        )
        return final_carry[5], final_carry[6]

    eval_keys = jax.random.split(rng, num_test_episodes)
    returns, lengths = jax.vmap(_run_one_episode)(eval_keys)
    return returns.mean(), lengths.mean()


def make_train(config: dict):
    env = FindGoal(
        num_agents=config["num_agents"],
        grid_size=config["grid_size"],
        max_steps=config["max_steps"],
        view_size=config["view_size"],
        view_tile_size=config["view_tile_size"],
        clutter_density=config["clutter_density"],
    )
    num_agents = env.num_agents
    action_dim = NUM_ACTIONS
    comm_len = config["comm_len"]
    hidden_size = config["gru_hidden_size"]
    num_envs = config["num_envs"]
    num_test_episodes = config.get("num_test_episodes", 5)
    max_steps = config["max_steps"]
    grid_size = config["grid_size"]
    img_feat_dim = config["img_feat_dim"]
    pos_feat_dim = config.get("pos_feat_dim", img_feat_dim)
    feat_dim = img_feat_dim + pos_feat_dim

    pov_shape = (env.view_px, env.view_px, 3)
    gs_dim = num_agents * (int(np.prod(pov_shape)) + 2)

    def train(rng: PRNGKeyArray, exp_id: int):
        def train_setup(rng: PRNGKeyArray):
            rng, _rng_reset = jax.random.split(rng)
            reset_keys = jax.random.split(_rng_reset, num_envs)
            obs, env_state = jax.vmap(env.reset)(reset_keys)
            selfpos = env_state.agent_pos

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
            actor = AECommActor(
                action_dim=action_dim,
                num_agents=num_agents,
                grid_size=grid_size,
                img_feat_dim=img_feat_dim,
                pos_feat_dim=pos_feat_dim,
                comm_len=comm_len,
                hidden_size=hidden_size,
                fc_dim_size=config["fc_dim_size"],
                activation=config["activation"],
            )
            critic = CriticDiscreteRNN(
                hidden_size=hidden_size,
                fc_dim_size=config["fc_dim_size"],
                activation=config["activation"],
            )

            ac_init_h = ScannedRNN.initialize_carry(num_agents * num_envs, hidden_size)
            cr_init_h = ScannedRNN.initialize_carry(num_envs, hidden_size)

            dummy_pov = jnp.zeros((1, *pov_shape), dtype=jnp.float32)
            dummy_sp = jnp.zeros((1, 2), dtype=jnp.int32)
            dummy_prev = jnp.zeros((1, num_agents, comm_len), dtype=jnp.float32)
            dummy_done = jnp.zeros((1,), dtype=bool)

            actor_ts = TrainState.create(
                apply_fn=actor.apply,
                params=actor.init(
                    rng_actor,
                    ac_init_h[:1],
                    dummy_pov,
                    dummy_sp,
                    dummy_prev,
                    0,
                    dummy_done,
                ),
                tx=tx_actor,
            )
            critic_ts = TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(
                    rng_critic,
                    cr_init_h,
                    (
                        jnp.zeros((1, num_envs, gs_dim)),
                        jnp.zeros((1, num_envs), dtype=bool),
                    ),
                ),
                tx=tx_critic,
            )

            actor_hs = ac_init_h.reshape(num_agents, num_envs, hidden_size)
            critic_hs = cr_init_h.reshape(num_envs, hidden_size)
            prev_msgs = jnp.zeros((num_envs, num_agents, comm_len), dtype=jnp.float32)
            return obs, env_state, selfpos, actor_ts, critic_ts, actor_hs, critic_hs, prev_msgs

        rng, _rng_setup = jax.random.split(rng)
        obs, env_state, selfpos, actor_ts, critic_ts, actor_hs, critic_hs, prev_msgs = (
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
            advantages, targets = _calculate_gae(
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
                a_ts = update_state.actor_train_state
                c_ts = update_state.critic_train_state

                rng_e, _rng_perm = jax.random.split(rng_e)
                permutation = jax.random.permutation(_rng_perm, num_envs)
                shuffled = (
                    jnp.take(init_actor_hs, permutation, axis=1),
                    init_critic_hs[permutation],
                    jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), traj_b),
                    jnp.take(adv, permutation, axis=1),
                    jnp.take(tgt, permutation, axis=1),
                )
                traj_shuf = shuffled[2]
                num_mb = config["num_minibatches"]
                mb_actor_h = shuffled[0].reshape(num_agents, num_mb, -1, hidden_size).swapaxes(
                    0, 1
                )
                mb_critic_h = shuffled[1].reshape(num_mb, -1, hidden_size)
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
                        logits, ae_losses = _actor_trajectory(
                            a_ts.apply_fn,
                            params,
                            init_h,
                            traj_mb.obs,
                            traj_mb.selfpos,
                            traj_mb.prev_msgs,
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
                        loss_actor = -jnp.minimum(loss_1, loss_2).mean()
                        entropy = distrax.Categorical(logits=logits).entropy().mean()
                        ae_mse = ae_losses.mean()
                        loss = (
                            loss_actor
                            - config["ent_coef"] * entropy
                            + config["recon_coef"] * ae_mse
                        )
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio - 1) - logratio).mean(),
                            "ae_mse": ae_mse,
                        }

                    def _critic_loss(params, init_h, traj_mb, tgt_mb):
                        def critic_scan(carry, inputs):
                            hs, gs, done_t = carry, inputs[0], inputs[1]
                            new_hs, values = _critic_values(
                                c_ts.apply_fn, params, hs, gs, done_t
                            )
                            return new_hs, values

                        _, values = jax.lax.scan(
                            critic_scan,
                            init_h,
                            (traj_mb.global_state, traj_mb.done),
                        )
                        values_b = values[:, :, None]
                        value_clipped = traj_mb.value + (
                            values_b - traj_mb.value
                        ).clip(-config["clip_eps"], config["clip_eps"])
                        per_agent = jnp.maximum(
                            jnp.square(values_b - tgt_mb),
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
                    (a_ts, c_ts),
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
                prev_msgs = runner_state.prev_msgs
                obs_a = runner_state.obs
                env_state = runner_state.env_state
                selfpos = env_state.agent_pos
                last_done = runner_state.done
                rng = runner_state.rng

                def reset_if_done(obs, state, prev_m, done_flag, key):
                    return jax.lax.cond(
                        done_flag,
                        lambda: (
                            *env.reset(key),
                            jnp.zeros((num_agents, comm_len), dtype=jnp.float32),
                        ),
                        lambda: (obs, state, prev_m),
                    )

                rng, _rng_reset = jax.random.split(rng)
                reset_keys = jax.random.split(_rng_reset, num_envs)
                obs_a, env_state, prev_msgs = jax.vmap(reset_if_done)(
                    obs_a,
                    env_state,
                    prev_msgs,
                    last_done,
                    reset_keys,
                )
                selfpos = env_state.agent_pos
                agent_done = env_state.agent_done
                global_state = _global_state(obs_a, selfpos)

                actor_hs, logits, msgs, ae_losses = _actor_rollout_step(
                    actor_ts.apply_fn,
                    actor_ts.params,
                    actor_hs,
                    obs_a,
                    selfpos,
                    prev_msgs,
                    last_done,
                    num_agents,
                )

                critic_hs, values = _critic_values(
                    critic_ts.apply_fn,
                    critic_ts.params,
                    critic_hs,
                    global_state,
                    last_done,
                )

                rng, _rng_action = jax.random.split(rng)
                action_keys = jax.random.split(
                    _rng_action, num_agents * num_envs
                ).reshape(num_agents, num_envs, -1)

                def sample_actions(keys, lg):
                    return jax.vmap(
                        lambda k, l: distrax.Categorical(logits=l).sample(seed=k)
                    )(keys, lg)

                actions = jax.vmap(sample_actions, in_axes=(0, 0))(
                    action_keys, logits
                )
                actions = jnp.where(agent_done.T, NUM_ACTIONS - 1, actions)
                actions = actions.transpose(1, 0)

                log_probs = jax.vmap(
                    lambda lg, act: distrax.Categorical(logits=lg).log_prob(act)
                )(logits.transpose(1, 0, 2), actions)

                rng, _rng_step = jax.random.split(rng)
                step_keys = jax.random.split(_rng_step, num_envs)

                def step_one_env(key, state, actions_per_agent):
                    return env.step(key, state, actions_per_agent)[:4]

                new_obs, new_env_state, rewards, new_done = jax.vmap(
                    step_one_env, in_axes=(0, 0, 0)
                )(step_keys, env_state, actions)

                new_msgs = msgs.transpose(1, 0, 2)
                new_prev_msgs = jnp.where(
                    new_done[:, None, None],
                    jnp.zeros_like(new_msgs),
                    new_msgs,
                )

                timesteps = runner_state.timesteps + 1
                timesteps = jnp.where(new_done, 0, timesteps)

                value_broadcast = jnp.broadcast_to(
                    values[:, None], (num_envs, num_agents)
                )
                transition = Transition(
                    obs=obs_a,
                    selfpos=selfpos,
                    prev_msgs=prev_msgs,
                    global_state=global_state,
                    action=actions,
                    log_prob=log_probs,
                    reward=rewards,
                    done=last_done,
                    new_done=new_done,
                    value=value_broadcast,
                    ae_loss=ae_losses.transpose(1, 0),
                )

                return (
                    RunnerState(
                        actor_train_state=actor_ts,
                        critic_train_state=critic_ts,
                        actor_hidden_states=actor_hs,
                        critic_hidden_states=critic_hs,
                        prev_msgs=new_prev_msgs,
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

            gs_last = _global_state(
                runner_state.obs, runner_state.env_state.agent_pos
            )
            _, last_values = _critic_values(
                runner_state.critic_train_state.apply_fn,
                runner_state.critic_train_state.params,
                runner_state.critic_hidden_states,
                gs_last,
                runner_state.done,
            )

            rng, _rng_update = jax.random.split(runner_state.rng)
            actor_ts, critic_ts, loss_info = _ppo_update(
                runner_state.actor_train_state,
                runner_state.critic_train_state,
                init_actor_hs,
                init_critic_hs,
                traj_batch,
                last_values,
                _rng_update,
            )
            loss_info = jax.tree.map(lambda x: x.mean(axis=(0, 1)), loss_info)

            rng, _rng_eval = jax.random.split(rng)
            test_return, test_episode_length = _eval_test_episodes(
                env,
                actor_ts.apply_fn,
                actor_ts.params,
                _rng_eval,
                num_agents,
                hidden_size,
                comm_len,
                max_steps,
                num_test_episodes,
            )

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
                "grad_norm_actor": loss_info["grad_norm_actor"],
                "grad_norm_critic": loss_info["grad_norm_critic"],
                "ae_mse": loss_info["ae_mse"],
                "test_return": test_return,
                "test_episode_length": test_episode_length,
            }

            def callback(exp_id, metric_dict):
                if LOGGER is not None:
                    LOGGER.log(
                        int(exp_id), {k: np.array(v) for k, v in metric_dict.items()}
                    )

            jax.experimental.io_callback(callback, None, exp_id, metric)

            return (
                RunnerState(
                    actor_train_state=actor_ts,
                    critic_train_state=critic_ts,
                    actor_hidden_states=runner_state.actor_hidden_states,
                    critic_hidden_states=runner_state.critic_hidden_states,
                    prev_msgs=runner_state.prev_msgs,
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
            actor_train_state=actor_ts,
            critic_train_state=critic_ts,
            actor_hidden_states=actor_hs,
            critic_hidden_states=critic_hs,
            prev_msgs=prev_msgs,
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


@hydra.main(version_base=None, config_path=".", config_name="config_mappo_ae_comm")
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

        print("Compiling MAPPO AEComm on FindGoal...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
