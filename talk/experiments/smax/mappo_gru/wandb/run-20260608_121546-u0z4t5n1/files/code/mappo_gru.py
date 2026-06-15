"""Team-shared MAPPO with GRU actor/critic on JaxMARL SMAX."""

import datetime
import functools
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper

from talk.networks.gru import CriticDiscreteRNN, ScannedRNN
from talk.networks.smax import ActorRNNAvailMasked
from talk.experiments.smax.env_utils import make_smax_env
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER: Optional[WandbMultiLogger] = None


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
    env = make_smax_env(config, scenario)

    num_agents = env.num_agents
    num_envs = int(config["num_envs"])
    num_steps = int(config["num_steps_per_env_per_update"])
    num_actors = num_agents * num_envs
    num_updates = int(config["total_timesteps"]) // num_steps // num_envs
    config["num_actors"] = num_actors
    config["num_updates"] = num_updates
    config["minibatch_size"] = num_actors * num_steps // int(config["num_minibatches"])

    clip_eps = float(config["clip_eps"])
    if config.get("scale_clip_eps", False):
        clip_eps = clip_eps / num_agents

    env = SMAXWorldStateWrapper(env, config["obs_with_agent_id"])
    env = SMAXLogWrapper(env)

    actor_lr = float(config["lr_actor"])
    critic_lr = float(config["lr_critic"])

    def _actor_lr_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["num_epochs"])) / num_updates
        return actor_lr * frac

    def _critic_lr_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["num_epochs"])) / num_updates
        return critic_lr * frac

    def train(rng: jnp.ndarray, exp_id: int):
        actor_network = ActorRNNAvailMasked(
            action_dim=env.action_space(env.agents[0]).n,
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            activation=config["activation"],
        )
        critic_network = CriticDiscreteRNN(
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            activation=config["activation"],
        )

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, num_envs, env.observation_space(env.agents[0]).shape[0]), dtype=jnp.float32),
            jnp.zeros((1, num_envs), dtype=bool),
            jnp.ones((num_envs, env.action_space(env.agents[0]).n), dtype=jnp.float32),
        )
        ac_init_h = ScannedRNN.initialize_carry(num_envs, config["gru_hidden_size"])
        actor_params = actor_network.init(rng_actor, ac_init_h, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, num_envs, env.world_state_size()), dtype=jnp.float32),
            jnp.zeros((1, num_envs), dtype=bool),
        )
        cr_init_h = ScannedRNN.initialize_carry(num_envs, config["gru_hidden_size"])
        critic_params = critic_network.init(rng_critic, cr_init_h, cr_init_x)

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
        ac_hstate = ScannedRNN.initialize_carry(num_actors, config["gru_hidden_size"])
        cr_hstate = ScannedRNN.initialize_carry(num_actors, config["gru_hidden_size"])

        def _update_step(update_runner_state, _):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, _):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                rng, rng_act = jax.random.split(rng)
                avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = batchify(avail, env.agents, num_actors)
                obs_batch = batchify(last_obs, env.agents, num_actors)
                ac_in = (obs_batch[None, :], last_done[None, :], avail)
                ac_h, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action.squeeze(), env.agents, num_envs, env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                world_state = last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))
                cr_in = (world_state[None, :], last_done[None, :])
                cr_h, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                rng, rng_step = jax.random.split(rng)
                step_rng = jax.random.split(rng_step, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(step_rng, env_state, env_act)
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
                runner_state = (train_states, env_state, obsv, done_batch, (ac_h, cr_h), rng)
                return runner_state, transition

            init_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, num_steps)
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))
            cr_in = (last_world_state[None, :], last_done[None, :])
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(trajectory, last_value):
                def _scan_gae(carry, transition):
                    gae, next_value = carry
                    done = transition.global_done
                    value = transition.value
                    reward = transition.reward
                    delta = reward + config["gamma"] * next_value * (1.0 - done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1.0 - done) * gae
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

            def _update_epoch(update_state, _):
                def _update_minbatch(train_states, batch_info):
                    actor_state, critic_state = train_states
                    ac_h, cr_h, trajectory, gae, target = batch_info

                    def _actor_loss(params, init_h, mb_traj, mb_gae):
                        _, pi = actor_network.apply(
                            params, init_h.squeeze(), (mb_traj.obs, mb_traj.done, mb_traj.avail_actions)
                        )
                        log_prob = pi.log_prob(mb_traj.action)
                        logratio = log_prob - mb_traj.log_prob
                        ratio = jnp.exp(logratio)
                        norm_gae = (mb_gae - mb_gae.mean()) / (mb_gae.std() + 1e-8)
                        loss1 = ratio * norm_gae
                        loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_gae
                        loss_actor = -jnp.minimum(loss1, loss2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)
                        actor_loss = loss_actor - config["ent_coef"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss(params, init_h, mb_traj, mb_targets):
                        _, value = critic_network.apply(
                            params, init_h.squeeze(), (mb_traj.world_state, mb_traj.done)
                        )
                        clipped = mb_traj.value + (value - mb_traj.value).clip(-clip_eps, clip_eps)
                        loss_v = jnp.square(value - mb_targets)
                        loss_v_clipped = jnp.square(clipped - mb_targets)
                        value_loss = 0.5 * jnp.maximum(loss_v, loss_v_clipped).mean()
                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    actor_out, actor_grads = actor_grad_fn(actor_state.params, ac_h, trajectory, gae)
                    critic_grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
                    critic_out, critic_grads = critic_grad_fn(critic_state.params, cr_h, trajectory, target)

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

                train_states, init_hstates, trajectory, gae, target, rng = update_state
                rng, rng_perm = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, num_actors, -1)), init_hstates)
                batch = (init_hstates[0], init_hstates[1], trajectory, gae.squeeze(), target.squeeze())
                permutation = jax.random.permutation(rng_perm, num_actors)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:])),
                        1,
                        0,
                    ),
                    shuffled,
                )
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    trajectory,
                    gae,
                    target,
                    rng,
                )
                return update_state, loss_info

            update_state = (train_states, init_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["num_epochs"])
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = jax.tree.map(
                lambda x: x.reshape((num_steps, num_envs, env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            def _log_callback(metric, exp_id):
                episode_mask = metric["returned_episode"][:, :, 0]
                returns = metric["returned_episode_returns"][:, :, 0][episode_mask].mean()
                win_rate = metric["returned_won_episode"][:, :, 0][episode_mask].mean()
                env_step = metric["update_steps"] * num_envs * num_steps
                payload = {
                    "returns": returns,
                    "win_rate": win_rate,
                    "env_step": env_step,
                    **metric["loss"],
                }
                if LOGGER is not None:
                    log_payload = {
                        k: float(np.asarray(v)) for k, v in payload.items()
                    }
                    LOGGER.log(
                        int(exp_id),
                        log_payload,
                        step=int(np.asarray(env_step)),
                    )

            metric["update_steps"] = update_step
            if config["use_wandb"]:
                jax.experimental.io_callback(_log_callback, None, metric, exp_id)
            update_step = update_step + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_step), metric

        rng, rng_runner = jax.random.split(rng)
        runner_state = (
            (actor_state, critic_state),
            env_state,
            obsv,
            jnp.zeros((num_actors,), dtype=bool),
            (ac_hstate, cr_hstate),
            rng_runner,
        )
        runner_state, _ = jax.lax.scan(_update_step, (runner_state, 0), None, num_updates)
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_mappo_gru")
def main(config):
    global LOGGER
    try:
        config = OmegaConf.to_container(config)
        group = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["use_wandb"]:
            LOGGER = WandbMultiLogger(
                project=config["project"],
                group=group,
                job_type=config["algorithm"] + config["custom_name"],
                config=config,
                mode="online",
                seed=config["seed"],
                num_seeds=config["num_seeds"],
            )
        else:
            LOGGER = None
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Compiling MAPPO-GRU SMAX...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
