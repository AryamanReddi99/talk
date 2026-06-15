"""MAPPO on MessageBox: decentralized actors, state-conditioned critics per agent."""

import datetime
from typing import Any, NamedTuple, Optional

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from talk.environments.message_box import NUM_MESSAGE_BITS, MessageBox
from talk.networks.mlp import ActorDiscrete, CriticDiscrete
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


class RunnerState(NamedTuple):
    actor_train_states: tuple
    critic_train_states: tuple
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
    traj_batch: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def _global_state(obs: FloatArray) -> FloatArray:
    """Flatten joint observations into a centralized state vector."""
    return obs.reshape(obs.shape[:-2] + (-1,))


def _slice_traj_per_agent(traj: Transition, agent_idx: int) -> Transition:
    return Transition(
        obs=traj.obs[:, :, agent_idx],
        global_state=traj.global_state,
        action=traj.action[:, :, agent_idx],
        log_prob=traj.log_prob[:, :, agent_idx],
        reward=traj.reward[:, :, agent_idx],
        done=traj.done[:, :, agent_idx],
        new_done=traj.new_done[:, :, agent_idx],
        value=traj.value[:, :, agent_idx],
    )


def make_train(config: dict):
    env = MessageBox(
        num_agents=config["num_agents"],
        max_steps=config["max_steps"],
        reward_correct=config["reward_correct"],
        reward_wrong=config["reward_wrong"],
    )
    num_agents = env.num_agents
    action_dims = [env.action_space(i).n for i in range(num_agents)]
    obs_dim = NUM_MESSAGE_BITS
    global_state_dim = num_agents * obs_dim

    config["batch_shuffle_dim"] = config["num_steps_per_env_per_update"] * config["num_envs"]

    def train(rng: PRNGKeyArray, exp_id: int):
        def train_setup(rng: PRNGKeyArray):
            rng, _rng_reset = jax.random.split(rng)
            reset_keys = jax.random.split(_rng_reset, config["num_envs"])

            def _reset(key):
                return env.reset(key)

            obs, env_state = jax.vmap(_reset)(reset_keys)

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

            def init_agent(agent_idx: int, rng_a: PRNGKeyArray):
                rng_a, rng_actor, rng_critic = jax.random.split(rng_a, 3)
                actor = ActorDiscrete(
                    action_dim=action_dims[agent_idx],
                    activation=config["activation"],
                    hidden_dim=config["fc_dim_size"],
                )
                critic = CriticDiscrete(
                    activation=config["activation"],
                    hidden_dim=config["fc_dim_size"],
                )
                local_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
                global_state = jnp.zeros((global_state_dim,), dtype=jnp.float32)
                actor_ts = TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(rng_actor, local_obs),
                    tx=tx_actor,
                )
                critic_ts = TrainState.create(
                    apply_fn=critic.apply,
                    params=critic.init(rng_critic, global_state),
                    tx=tx_critic,
                )
                return actor_ts, critic_ts

            rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
            actor_states = []
            critic_states = []
            for i in range(num_agents):
                a_ts, c_ts = init_agent(i, agent_rngs[i])
                actor_states.append(a_ts)
                critic_states.append(c_ts)
            return obs, env_state, tuple(actor_states), tuple(critic_states)

        rng, _rng_setup = jax.random.split(rng)
        obs, env_state, actor_train_states, critic_train_states = train_setup(_rng_setup)

        def _ppo_update_agent(
            actor_ts: TrainState,
            critic_ts: TrainState,
            traj: Transition,
            last_val: FloatArray,
            rng: PRNGKeyArray,
        ):
            def _calculate_gae(traj_batch: Transition, last_v: FloatArray):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done = transition.new_done
                    value = transition.value
                    reward = transition.reward
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_v), last_v),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj, last_val)
            update_state = UpdateState(
                actor_train_state=actor_ts,
                critic_train_state=critic_ts,
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
                batch = (traj_b, adv, tgt)

                def _reshape(x):
                    if x.ndim == 2:
                        return x.reshape(-1)
                    if x.ndim == 3:
                        return x.reshape(-1, *x.shape[2:])
                    return x.reshape(-1, *x.shape[3:])

                batch_flat = jax.tree.map(_reshape, batch)
                perm = jax.random.permutation(_rng_perm, config["batch_shuffle_dim"])
                batch_shuf = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch_flat)
                minibatches = jax.tree.map(
                    lambda x: x.reshape(config["num_minibatches"], -1, *x.shape[1:]),
                    batch_shuf,
                )

                def _update_minibatch(carry, minibatch):
                    a_ts, c_ts = carry
                    traj_mb, adv_mb, tgt_mb = minibatch

                    def _actor_loss(params, traj_mb, adv_mb):
                        pi = a_ts.apply_fn(params, traj_mb.obs)
                        log_prob = pi.log_prob(traj_mb.action)
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
                        entropy = pi.entropy().mean()
                        loss = loss_actor - config["ent_coef"] * entropy
                        return loss, {
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "approx_kl": ((ratio - 1) - logratio).mean(),
                        }

                    def _critic_loss(params, traj_mb, tgt_mb):
                        value = c_ts.apply_fn(params, traj_mb.global_state)
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
                    )(a_ts.params, traj_mb, adv_mb)
                    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                        _critic_loss, has_aux=True
                    )(c_ts.params, traj_mb, tgt_mb)

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

            def _env_step(runner_state: RunnerState, _):
                actor_states = runner_state.actor_train_states
                critic_states = runner_state.critic_train_states
                obs_a = runner_state.obs
                env_state = runner_state.env_state
                done = runner_state.done
                rng = runner_state.rng
                global_state = _global_state(obs_a)

                def reset_if_done(obs, state, done_flag, key):
                    def _do_reset():
                        return env.reset(key)

                    def _keep():
                        return obs, state

                    return jax.lax.cond(done_flag, _do_reset, _keep)

                rng, _rng_reset = jax.random.split(rng)
                reset_keys = jax.random.split(_rng_reset, config["num_envs"])
                obs_a, env_state = jax.vmap(reset_if_done)(obs_a, env_state, done, reset_keys)
                global_state = _global_state(obs_a)

                rng, _rng_action = jax.random.split(rng)
                action_keys = jax.random.split(
                    _rng_action, num_agents * config["num_envs"]
                ).reshape(num_agents, config["num_envs"], -1)

                def act_one_env(local_obs, global_s, key_e, actor_ts, critic_ts):
                    pi = actor_ts.apply_fn(actor_ts.params, local_obs)
                    value = critic_ts.apply_fn(critic_ts.params, global_s)
                    action = pi.sample(seed=key_e)
                    return action, pi.log_prob(action), value

                actions_list = []
                log_probs_list = []
                values_list = []
                for i in range(num_agents):
                    a, lp, v = jax.vmap(act_one_env, in_axes=(0, 0, 0, None, None))(
                        obs_a[:, i],
                        global_state,
                        action_keys[i],
                        actor_states[i],
                        critic_states[i],
                    )
                    actions_list.append(a)
                    log_probs_list.append(lp)
                    values_list.append(v)
                actions = jnp.stack(actions_list)
                log_probs = jnp.stack(log_probs_list)
                values = jnp.stack(values_list)

                rng, _rng_step = jax.random.split(rng)
                step_keys = jax.random.split(_rng_step, config["num_envs"])

                def step_one_env(key, state, actions_per_agent):
                    new_obs, new_state, rewards, step_done, _ = env.step(
                        key, state, actions_per_agent
                    )
                    return new_obs, new_state, rewards, step_done

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
                    done=jnp.broadcast_to(done[:, None], (config["num_envs"], num_agents)),
                    new_done=jnp.broadcast_to(new_done[:, None], (config["num_envs"], num_agents)),
                    value=values.transpose(1, 0),
                )

                return (
                    RunnerState(
                        actor_train_states=actor_states,
                        critic_train_states=critic_states,
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
            last_values = []
            for i in range(num_agents):
                c_ts = runner_state.critic_train_states[i]
                last_values.append(c_ts.apply_fn(c_ts.params, global_state_last))
            last_values = jnp.stack(last_values)

            rng, *agent_rngs = jax.random.split(runner_state.rng, num_agents + 1)
            new_actor_states = []
            new_critic_states = []
            loss_infos = []
            for i in range(num_agents):
                traj_a = _slice_traj_per_agent(traj_batch, i)
                new_a, new_c, loss_i = _ppo_update_agent(
                    runner_state.actor_train_states[i],
                    runner_state.critic_train_states[i],
                    traj_a,
                    last_values[i],
                    agent_rngs[i],
                )
                new_actor_states.append(new_a)
                new_critic_states.append(new_c)
                loss_infos.append(loss_i)
            actor_train_states = tuple(new_actor_states)
            critic_train_states = tuple(new_critic_states)
            loss_info = jax.tree.map(lambda *xs: jnp.stack(xs), *loss_infos)

            reward = traj_batch.reward
            done = traj_batch.new_done[:, :, 0]
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

            mean_reward_per_step = reward.mean(axis=(0, 1))

            (new_cumulative_return, _), (ret_at_done, len_at_done) = jax.lax.scan(
                _rollout_episode_metrics,
                (runner_state.cumulative_return, initial_timesteps),
                (reward_agent0, done),
            )
            num_episodes = done.sum()
            returns_avg = jnp.where(num_episodes > 0, ret_at_done.sum() / num_episodes, 0.0)
            episode_length_avg = jnp.where(num_episodes > 0, len_at_done.sum() / num_episodes, 0.0)

            loss_info = jax.tree.map(lambda x: x.mean(axis=(0, 1)), loss_info)

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
                "mean_reward_per_step": mean_reward_per_step.mean(),
            }
            for i in range(num_agents):
                metric[f"return_agent_{i}"] = jnp.where(
                    num_episodes > 0,
                    ret_at_done.sum() / num_episodes,
                    0.0,
                )
                metric[f"mean_reward_agent_{i}"] = reward[:, :, i].mean()
                metric[f"actor_loss_agent_{i}"] = loss_info["actor_loss"][i]
                metric[f"value_loss_agent_{i}"] = loss_info["value_loss"][i]
                metric[f"entropy_agent_{i}"] = loss_info["entropy"][i]
                metric[f"grad_norm_actor_agent_{i}"] = loss_info["grad_norm_actor"][i]
                metric[f"grad_norm_critic_agent_{i}"] = loss_info["grad_norm_critic"][i]

            metric["actor_loss"] = loss_info["actor_loss"].mean()
            metric["value_loss"] = loss_info["value_loss"].mean()
            metric["entropy"] = loss_info["entropy"].mean()
            metric["approx_kl"] = loss_info["approx_kl"].mean()
            metric["total_loss"] = loss_info["total_loss"].mean()
            metric["grad_norm_actor"] = loss_info["grad_norm_actor"].mean()
            metric["grad_norm_critic"] = loss_info["grad_norm_critic"].mean()

            def callback(exp_id, metric_dict):
                if LOGGER is not None:
                    LOGGER.log(int(exp_id), {k: np.array(v) for k, v in metric_dict.items()})

            jax.experimental.io_callback(callback, None, exp_id, metric)

            return (
                RunnerState(
                    actor_train_states=actor_train_states,
                    critic_train_states=critic_train_states,
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
            actor_train_states=actor_train_states,
            critic_train_states=critic_train_states,
            env_state=env_state,
            obs=obs,
            done=jnp.zeros((config["num_envs"],), dtype=jnp.bool_),
            cumulative_return=jnp.zeros((config["num_envs"],), dtype=jnp.float32),
            timesteps=jnp.zeros((config["num_envs"],), dtype=jnp.int32),
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

        print("Compiling MAPPO...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
