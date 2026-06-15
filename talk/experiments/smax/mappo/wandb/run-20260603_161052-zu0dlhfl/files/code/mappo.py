"""Team-shared feedforward MAPPO on JaxMARL SMAX."""

import datetime
import functools
from typing import Any, Dict, NamedTuple, Optional, Sequence

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper

from talk.networks.mlp import CriticDiscrete
from talk.networks.smax import ActorDiscreteAvailMasked

LOGGER: Optional[wandb.sdk.wandb_run.Run] = None


class SMAXWorldStateWrapper:
    """Adds world-state observations per ally for centralized critic."""

    def __init__(self, env: HeuristicEnemySMAX, obs_with_agent_id: bool = True):
        self._env = env
        self.obs_with_agent_id = bool(obs_with_agent_id)
        if self.obs_with_agent_id:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self._ws_with_agent_id
        else:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self._ws_only

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs)
        return obs, env_state

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs)
        return obs, env_state, reward, done, info

    @functools.partial(jax.jit, static_argnums=0)
    def _ws_only(self, obs):
        world_state = obs["world_state"]
        return world_state[None].repeat(self._env.num_allies, axis=0)

    @functools.partial(jax.jit, static_argnums=0)
    def _ws_with_agent_id(self, obs):
        world_state = obs["world_state"][None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies, dtype=world_state.dtype)
        return jnp.concatenate((world_state, one_hot), axis=-1)

    def world_state_size(self) -> int:
        return int(self._world_state_size)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: Dict[str, jnp.ndarray]
    avail_actions: jnp.ndarray


class MiniBatch(NamedTuple):
    obs: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    old_value: jnp.ndarray
    old_log_prob: jnp.ndarray
    world_state: jnp.ndarray
    avail_actions: jnp.ndarray
    gae: jnp.ndarray
    targets: jnp.ndarray


def batchify(x: dict, agent_list: Sequence[str], num_actors: int) -> jnp.ndarray:
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(
    x: jnp.ndarray, agent_list: Sequence[str], num_envs: int, num_actors: int
) -> dict:
    x = x.reshape((num_actors, num_envs, -1))
    return {agent: x[i] for i, agent in enumerate(agent_list)}


def make_train(config: Dict[str, Any]):
    scenario = map_name_to_scenario(config["map_name"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["env_kwargs"])

    num_agents = env.num_agents
    num_envs = int(config["num_envs"])
    num_steps = int(config["num_steps_per_env_per_update"])
    num_actors = num_agents * num_envs
    num_updates = int(config["total_timesteps"]) // num_steps // num_envs
    config["num_actors"] = num_actors
    config["num_updates"] = num_updates
    batch_size = num_actors * num_steps
    minibatch_size = batch_size // int(config["num_minibatches"])

    clip_eps = float(config["clip_eps"])
    if config.get("scale_clip_eps", False):
        clip_eps = clip_eps / num_agents

    env = SMAXWorldStateWrapper(env, config["obs_with_agent_id"])
    env = SMAXLogWrapper(env)

    actor_lr = float(config["lr_actor"])
    critic_lr = float(config["lr_critic"])

    def _actor_lr_schedule(count):
        frac = (
            1.0
            - (count // (config["num_minibatches"] * config["num_epochs"]))
            / num_updates
        )
        return actor_lr * frac

    def _critic_lr_schedule(count):
        frac = (
            1.0
            - (count // (config["num_minibatches"] * config["num_epochs"]))
            / num_updates
        )
        return critic_lr * frac

    def train(rng: jnp.ndarray):
        action_dim = env.action_space(env.agents[0]).n
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        world_state_dim = env.world_state_size()

        actor_network = ActorDiscreteAvailMasked(
            action_dim=action_dim,
            activation=config["activation"],
            fc_dim_size=config["fc_dim_size"],
        )
        critic_network = CriticDiscrete(
            activation=config["activation"],
            hidden_dim=config["fc_dim_size"],
        )

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        actor_params = actor_network.init(
            rng_actor,
            jnp.zeros((num_actors, obs_dim), dtype=jnp.float32),
            jnp.ones((num_actors, action_dim), dtype=jnp.float32),
        )
        critic_params = critic_network.init(
            rng_critic, jnp.zeros((num_actors, world_state_dim), dtype=jnp.float32)
        )

        if config["anneal_lr"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=_actor_lr_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=_critic_lr_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=actor_lr, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=critic_lr, eps=1e-5),
            )

        actor_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

        rng, rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(rng_reset, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        def _update_step(update_runner_state, _):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, _):
                train_states, env_state, last_obs, last_done, rng = runner_state

                rng, rng_act = jax.random.split(rng)
                avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = batchify(avail, env.agents, num_actors)
                obs_batch = batchify(last_obs, env.agents, num_actors)
                pi = actor_network.apply(train_states[0].params, obs_batch, avail)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action.squeeze(), env.agents, num_envs, env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                world_state = (
                    last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))
                )
                value = critic_network.apply(train_states[1].params, world_state)

                rng, rng_step = jax.random.split(rng)
                step_rng = jax.random.split(rng_step, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    step_rng, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                done_batch = batchify(done, env.agents, num_actors).squeeze()
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, num_actors).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    world_state=world_state,
                    info=info,
                    avail_actions=avail,
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )
            train_states, env_state, last_obs, last_done, rng = runner_state

            last_world_state = (
                last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))
            )
            last_val = critic_network.apply(
                train_states[1].params, last_world_state
            ).squeeze()

            def _calculate_gae(trajectory, last_value):
                def _scan_gae(carry, transition):
                    gae, next_value = carry
                    done = transition.global_done
                    value = transition.value
                    reward = transition.reward
                    delta = reward + config["gamma"] * next_value * (1.0 - done) - value
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1.0 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _scan_gae,
                    (jnp.zeros_like(last_value), last_value),
                    trajectory,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + trajectory.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _to_minibatches(rng, trajectory, gae, target):
                mb = MiniBatch(
                    obs=trajectory.obs.reshape((batch_size, -1)),
                    done=trajectory.done.reshape((batch_size,)),
                    action=trajectory.action.reshape((batch_size,)),
                    old_value=trajectory.value.reshape((batch_size,)),
                    old_log_prob=trajectory.log_prob.reshape((batch_size,)),
                    world_state=trajectory.world_state.reshape((batch_size, -1)),
                    avail_actions=trajectory.avail_actions.reshape((batch_size, -1)),
                    gae=gae.reshape((batch_size,)),
                    targets=target.reshape((batch_size,)),
                )
                permutation = jax.random.permutation(rng, batch_size)
                mb = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), mb)
                mb = jax.tree.map(
                    lambda x: x.reshape(
                        (config["num_minibatches"], minibatch_size) + x.shape[1:]
                    ),
                    mb,
                )
                return mb

            def _update_epoch(update_state, _):
                actor_state, critic_state, rng = update_state
                rng, rng_perm = jax.random.split(rng)
                minibatches = _to_minibatches(rng_perm, traj_batch, advantages, targets)

                def _update_minibatch(train_states, minibatch):
                    actor_state, critic_state = train_states

                    def _actor_loss(params):
                        pi = actor_network.apply(
                            params, minibatch.obs, minibatch.avail_actions
                        )
                        log_prob = pi.log_prob(minibatch.action)
                        logratio = log_prob - minibatch.old_log_prob
                        ratio = jnp.exp(logratio)
                        norm_gae = (minibatch.gae - minibatch.gae.mean()) / (
                            minibatch.gae.std() + 1e-8
                        )
                        loss1 = ratio * norm_gae
                        loss2 = (
                            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_gae
                        )
                        loss_actor = -jnp.minimum(loss1, loss2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
                        actor_loss = loss_actor - config["ent_coef"] * entropy
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss(params):
                        value = critic_network.apply(params, minibatch.world_state)
                        clipped = minibatch.old_value + (
                            value - minibatch.old_value
                        ).clip(-clip_eps, clip_eps)
                        loss_v = jnp.square(value - minibatch.targets)
                        loss_v_clipped = jnp.square(clipped - minibatch.targets)
                        value_loss = 0.5 * jnp.maximum(loss_v, loss_v_clipped).mean()
                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    actor_out, actor_grads = actor_grad_fn(actor_state.params)
                    critic_grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
                    critic_out, critic_grads = critic_grad_fn(critic_state.params)

                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    loss_info = {
                        "total_loss": actor_out[0] + critic_out[0],
                        "actor_loss": actor_out[0],
                        "value_loss": critic_out[0],
                        "entropy": actor_out[1][1],
                        "ratio": actor_out[1][2],
                        "approx_kl": actor_out[1][3],
                        "clip_frac": actor_out[1][4],
                    }
                    return (actor_state, critic_state), loss_info

                (actor_state, critic_state), loss_info = jax.lax.scan(
                    _update_minibatch, (actor_state, critic_state), minibatches
                )
                return (actor_state, critic_state, rng), loss_info

            (actor_state, critic_state, rng), loss_info = jax.lax.scan(
                _update_epoch,
                (train_states[0], train_states[1], rng),
                None,
                config["num_epochs"],
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            metric = jax.tree.map(
                lambda x: x.reshape((num_steps, num_envs, env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info

            def _log_callback(metric):
                episode_mask = metric["returned_episode"][:, :, 0]
                returns = metric["returned_episode_returns"][:, :, 0][
                    episode_mask
                ].mean()
                win_rate = metric["returned_won_episode"][:, :, 0][episode_mask].mean()
                env_step = metric["update_steps"] * num_envs * num_steps
                payload = {
                    "returns": returns,
                    "win_rate": win_rate,
                    "env_step": env_step,
                    **metric["loss"],
                }
                if LOGGER is not None:
                    LOGGER.log(payload, step=int(env_step))

            metric["update_steps"] = update_step
            if config["use_wandb"]:
                jax.experimental.io_callback(_log_callback, None, metric)
            update_step = update_step + 1

            runner_state = (
                (actor_state, critic_state),
                env_state,
                last_obs,
                last_done,
                rng,
            )
            return (runner_state, update_step), metric

        rng, rng_runner = jax.random.split(rng)
        runner_state = (
            (actor_state, critic_state),
            env_state,
            obsv,
            jnp.zeros((num_actors,), dtype=bool),
            rng_runner,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, num_updates
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_mappo")
def main(config):
    global LOGGER
    try:
        config = OmegaConf.to_container(config)
        group = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["use_wandb"]:
            LOGGER = wandb.init(
                project=config["project"],
                group=group,
                job_type=config["algorithm"] + config["custom_name"],
                name=f"{config['seed']}_{group}",
                config=config,
                mode="online",
            )
        else:
            LOGGER = None
        rng = jax.random.PRNGKey(config["seed"])

        print("Compiling MAPPO SMAX...")
        train_fn = jax.jit(make_train(config))
        print("Running...")
        jax.block_until_ready(train_fn(rng))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
