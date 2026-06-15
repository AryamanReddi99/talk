"""Team-shared MAPPO + TarMAC communication on JaxMARL SMAX."""

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
from jaxmarl.wrappers.baselines import SMAXLogWrapper

from talk.networks.gru import CriticDiscreteRNN, ScannedRNN
from talk.networks.smax import ActorTarMACRNNAvailMasked
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
    ally_positions: jnp.ndarray


def ally_comm_reachability(positions: jnp.ndarray, comm_range: float) -> jnp.ndarray:
    """Ally positions (N, 2) in map units → (N, N) bool reachability matrix."""
    n = positions.shape[0]
    if comm_range < 0:
        return jnp.ones((n, n), dtype=bool)
    dist = jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return dist <= comm_range


def batchify(x: dict, agent_list: Sequence[str], num_actors: int) -> jnp.ndarray:
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(
    x: jnp.ndarray, agent_list: Sequence[str], num_envs: int, num_actors: int
) -> dict:
    x = x.reshape((num_actors, num_envs, -1))
    return {agent: x[i] for i, agent in enumerate(agent_list)}


def to_env_major(
    x: jnp.ndarray, num_envs: int, num_agents: int
) -> jnp.ndarray:
    """(num_actors, ...) -> (E, N, ...) — batchify order is agent-major."""
    rest = x.shape[1:]
    return x.reshape(num_agents, num_envs, *rest).swapaxes(0, 1)


def to_actor_major(
    x: jnp.ndarray, num_envs: int, num_agents: int
) -> jnp.ndarray:
    """(E, N, ...) -> (num_actors, ...)."""
    rest = x.shape[2:]
    return x.swapaxes(0, 1).reshape(num_agents * num_envs, *rest)


def traj_field_to_env_major(
    x: jnp.ndarray, num_envs: int, num_agents: int
) -> jnp.ndarray:
    """(T, num_actors, ...) -> (T, E, N, ...)."""
    if x.ndim == 2:
        return x.reshape(x.shape[0], num_agents, num_envs).swapaxes(1, 2)
    return x.reshape(x.shape[0], num_agents, num_envs, *x.shape[2:]).swapaxes(1, 2)


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
    config["minibatch_size"] = num_actors * num_steps // int(config["num_minibatches"])

    sig_dim = int(config["sig_dim"])
    val_dim = int(config["val_dim"])
    comm_range = float(config.get("comm_range", -1))
    scan_unroll = int(config.get("trajectory_scan_unroll", 8))
    num_minibatches = int(config["num_minibatches"])
    assert num_envs % num_minibatches == 0, "num_envs must divide num_minibatches"
    mb_envs = num_envs // num_minibatches

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

    def _tarmac_env_step(
        actor_apply,
        params,
        ac_h_flat: jnp.ndarray,
        obs_flat: jnp.ndarray,
        avail_flat: jnp.ndarray,
        done_flat: jnp.ndarray,
        prev_sig: jnp.ndarray,
        prev_val: jnp.ndarray,
        ally_pos: jnp.ndarray,
    ):
        """
        TarMAC forward for all envs.
        ac_h_flat / obs_flat / avail_flat / done_flat: (num_actors, ...)
        prev_sig / prev_val: (E, N, sig_dim) and (E, N, val_dim)
        ally_pos: (E, N, 2) ally positions at step start (map units)
        """
        h = to_env_major(ac_h_flat, num_envs, num_agents)
        obs = to_env_major(obs_flat, num_envs, num_agents)
        avail = to_env_major(avail_flat, num_envs, num_agents)
        done = to_env_major(done_flat, num_envs, num_agents)

        def _one_env(h_e, obs_e, avail_e, done_e, ps_e, pv_e, pos_e):
            reach = ally_comm_reachability(pos_e, comm_range)
            return actor_apply(
                params,
                h_e,
                obs_e,
                ps_e,
                pv_e,
                done_e,
                avail_e,
                reach,
                method=ActorTarMACRNNAvailMasked.step,
            )

        # vmapped over E: hidden (E, N, H), logits (E, N, A), etc.
        new_h, sig, val, logits = jax.vmap(_one_env)(
            h, obs, avail, done, prev_sig, prev_val, ally_pos
        )
        return (
            to_actor_major(new_h, num_envs, num_agents),
            sig,
            val,
            to_actor_major(logits, num_envs, num_agents),
        )

    def _actor_trajectory(
        actor_apply,
        params,
        init_h: jnp.ndarray,
        init_prev_sig: jnp.ndarray,
        init_prev_val: jnp.ndarray,
        traj,
        mb_envs: int,
    ):
        """
        Recompute TarMAC policy over a trajectory for PPO (messages not stored).

        init_h: (mb_E, N, H)
        init_prev_sig / init_prev_val: (mb_E, N, sig/val_dim) at rollout start
        traj.obs / done / avail: (T, mb_E, N, ...)
        returns pi: distrax.Categorical with logits (T, mb_E, N, A)
        """
        def _scan_step(carry, inputs):
            h, ps, pv = carry
            obs_t, done_t, avail_t, ally_pos_t, global_done_t = inputs

            def _one_env(h_e, obs_e, done_e, avail_e, ps_e, pv_e, pos_e):
                reach = ally_comm_reachability(pos_e, comm_range)
                return actor_apply(
                    params,
                    h_e,
                    obs_e,
                    ps_e,
                    pv_e,
                    done_e,
                    avail_e,
                    reach,
                    method=ActorTarMACRNNAvailMasked.step,
                )

            new_h, sig, val, logits = jax.vmap(_one_env)(
                h, obs_t, done_t, avail_t, ps, pv, ally_pos_t
            )
            # Match rollout: zero outgoing messages when the env episode ends.
            ep_done = global_done_t[:, 0:1, None]
            sig = jnp.where(ep_done, 0.0, sig)
            val = jnp.where(ep_done, 0.0, val)
            return (new_h, sig, val), logits

        _, logits = jax.lax.scan(
            _scan_step,
            (init_h, init_prev_sig, init_prev_val),
            (traj.obs, traj.done, traj.avail_actions, traj.ally_positions, traj.global_done),
            unroll=scan_unroll,
        )
        return distrax.Categorical(logits=logits)

    def train(rng: jnp.ndarray, exp_id: int):
        actor_network = ActorTarMACRNNAvailMasked(
            action_dim=env.action_space(env.agents[0]).n,
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            sig_dim=sig_dim,
            val_dim=val_dim,
            activation=config["activation"],
        )
        critic_network = CriticDiscreteRNN(
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            activation=config["activation"],
        )

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n
        ac_init_h = ScannedRNN.initialize_carry(num_agents, config["gru_hidden_size"])
        actor_params = actor_network.init(
            rng_actor,
            ac_init_h,
            jnp.zeros((num_agents, obs_dim)),
            jnp.zeros((num_agents, sig_dim)),
            jnp.zeros((num_agents, val_dim)),
            jnp.zeros((num_agents,), dtype=bool),
            jnp.ones((num_agents, action_dim)),
            jnp.ones((num_agents, num_agents), dtype=bool),
            method=ActorTarMACRNNAvailMasked.step,
        )

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
        # Outgoing TarMAC messages from t-1; shape (E, N, sig/val_dim)
        prev_sig = jnp.zeros((num_envs, num_agents, sig_dim))
        prev_val = jnp.zeros((num_envs, num_agents, val_dim))

        def _update_step(update_runner_state, _):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, _):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    hstates,
                    prev_sig_carry,
                    prev_val_carry,
                    rng,
                ) = runner_state

                rng, rng_act = jax.random.split(rng)
                avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = batchify(avail, env.agents, num_actors)
                obs_batch = batchify(last_obs, env.agents, num_actors)
                ally_pos = env_state.env_state.state.unit_positions[:, :num_agents, :]
                ac_h, sig, val, logits = _tarmac_env_step(
                    actor_network.apply,
                    train_states[0].params,
                    hstates[0],
                    obs_batch,
                    avail,
                    last_done,
                    prev_sig_carry,
                    prev_val_carry,
                    ally_pos,
                )
                pi = distrax.Categorical(logits=logits)
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
                global_done = done["__all__"]
                # Reset outgoing messages when the env episode ends; shape (E, 1, 1)
                ep_done = global_done[:, None, None]
                prev_sig_next = jnp.where(ep_done, 0.0, sig)
                prev_val_next = jnp.where(ep_done, 0.0, val)

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
                    ally_positions=to_actor_major(ally_pos, num_envs, num_agents),
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_h, cr_h),
                    prev_sig_next,
                    prev_val_next,
                    rng,
                )
                return runner_state, transition

            init_hstates = runner_state[4]
            prev_sig_init = runner_state[5]
            prev_val_init = runner_state[6]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, num_steps)
            train_states, env_state, last_obs, last_done, hstates, prev_sig, prev_val, rng = (
                runner_state
            )

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

            def _reshape_env_minibatches(x_env, perm):
                """(T, E, ...) -> (num_mb, T, mb_E, ...) via env shuffle perm (E,)."""
                x_shuf = x_env[:, perm]
                if x_env.ndim == 3:
                    return x_shuf.reshape(
                        x_shuf.shape[0], num_minibatches, mb_envs, x_shuf.shape[-1]
                    ).swapaxes(1, 0)
                return x_shuf.reshape(
                    x_shuf.shape[0],
                    num_minibatches,
                    mb_envs,
                    num_agents,
                    *x_shuf.shape[3:],
                ).swapaxes(1, 0)

            def _update_epoch(update_state, _):
                def _update_minbatch(train_states, batch_info):
                    actor_state, critic_state = train_states
                    ac_h, cr_h, ps_init, pv_init, traj_mb, gae_mb, target_mb = batch_info

                    def _actor_loss(params, init_h, ps0, pv0, mb_traj, mb_gae):
                        pi = _actor_trajectory(
                            actor_network.apply,
                            params,
                            init_h,
                            ps0,
                            pv0,
                            mb_traj,
                            mb_envs,
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
                        # Flatten (T, mb_E, N, ...) -> (T, mb_E*N, ...) for critic ScannedRNN
                        mb_actors = mb_envs * num_agents
                        ws = mb_traj.world_state.reshape(
                            mb_traj.world_state.shape[0], mb_actors, -1
                        )
                        done = mb_traj.done.reshape(mb_traj.done.shape[0], mb_actors)
                        # init_h: (mb_E, N, H) -> (mb_E*N, H)
                        init_h_flat = init_h.reshape(mb_actors, -1)
                        _, value = critic_network.apply(
                            params, init_h_flat, (ws, done)
                        )
                        value = value.reshape(mb_traj.value.shape)
                        clipped = mb_traj.value + (value - mb_traj.value).clip(-clip_eps, clip_eps)
                        loss_v = jnp.square(value - mb_targets)
                        loss_v_clipped = jnp.square(clipped - mb_targets)
                        value_loss = 0.5 * jnp.maximum(loss_v, loss_v_clipped).mean()
                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    actor_out, actor_grads = actor_grad_fn(
                        actor_state.params, ac_h, ps_init, pv_init, traj_mb, gae_mb
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
                    critic_out, critic_grads = critic_grad_fn(
                        critic_state.params, cr_h, traj_mb, target_mb
                    )

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

                train_states, init_hstates, prev_sig_init, prev_val_init, trajectory, gae, target, rng = update_state
                rng, rng_perm = jax.random.split(rng)

                # Env-major layouts for TarMAC (keep all N agents per env together).
                init_h_env = to_env_major(init_hstates[0], num_envs, num_agents)
                init_c_env = to_env_major(init_hstates[1], num_envs, num_agents)
                # Message carry at rollout start: (E, N, sig/val_dim)
                prev_sig_env = prev_sig_init
                prev_val_env = prev_val_init
                traj_env = Transition(
                    global_done=traj_field_to_env_major(trajectory.global_done, num_envs, num_agents),
                    done=traj_field_to_env_major(trajectory.done, num_envs, num_agents),
                    action=traj_field_to_env_major(trajectory.action, num_envs, num_agents),
                    value=traj_field_to_env_major(trajectory.value, num_envs, num_agents),
                    reward=traj_field_to_env_major(trajectory.reward, num_envs, num_agents),
                    log_prob=traj_field_to_env_major(trajectory.log_prob, num_envs, num_agents),
                    obs=traj_field_to_env_major(trajectory.obs, num_envs, num_agents),
                    world_state=traj_field_to_env_major(
                        trajectory.world_state, num_envs, num_agents
                    ),
                    info=trajectory.info,
                    avail_actions=traj_field_to_env_major(
                        trajectory.avail_actions, num_envs, num_agents
                    ),
                    ally_positions=traj_field_to_env_major(
                        trajectory.ally_positions, num_envs, num_agents
                    ),
                )
                gae_env = traj_field_to_env_major(gae, num_envs, num_agents)
                target_env = traj_field_to_env_major(target, num_envs, num_agents)

                perm = jax.random.permutation(rng_perm, num_envs)
                init_h_mb = init_h_env[perm].reshape(num_minibatches, mb_envs, num_agents, -1)
                init_c_mb = init_c_env[perm].reshape(num_minibatches, mb_envs, num_agents, -1)
                prev_sig_mb = prev_sig_env[perm].reshape(
                    num_minibatches, mb_envs, num_agents, sig_dim
                )
                prev_val_mb = prev_val_env[perm].reshape(
                    num_minibatches, mb_envs, num_agents, val_dim
                )
                traj_mb = Transition(
                    global_done=_reshape_env_minibatches(traj_env.global_done, perm),
                    done=_reshape_env_minibatches(traj_env.done, perm),
                    action=_reshape_env_minibatches(traj_env.action, perm),
                    value=_reshape_env_minibatches(traj_env.value, perm),
                    reward=_reshape_env_minibatches(traj_env.reward, perm),
                    log_prob=_reshape_env_minibatches(traj_env.log_prob, perm),
                    obs=_reshape_env_minibatches(traj_env.obs, perm),
                    world_state=_reshape_env_minibatches(traj_env.world_state, perm),
                    info={},  # logging only; not shuffled into PPO minibatches
                    avail_actions=_reshape_env_minibatches(traj_env.avail_actions, perm),
                    ally_positions=_reshape_env_minibatches(traj_env.ally_positions, perm),
                )
                gae_mb = _reshape_env_minibatches(gae_env, perm)
                target_mb = _reshape_env_minibatches(target_env, perm)

                minibatches = (init_h_mb, init_c_mb, prev_sig_mb, prev_val_mb, traj_mb, gae_mb, target_mb)
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    init_hstates,
                    prev_sig_init,
                    prev_val_init,
                    trajectory,
                    gae,
                    target,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                init_hstates,
                prev_sig_init,
                prev_val_init,
                traj_batch,
                advantages,
                targets,
                rng,
            )
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
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                prev_sig,
                prev_val,
                rng,
            )
            return (runner_state, update_step), metric

        rng, rng_runner = jax.random.split(rng)
        runner_state = (
            (actor_state, critic_state),
            env_state,
            obsv,
            jnp.zeros((num_actors,), dtype=bool),
            (ac_hstate, cr_hstate),
            prev_sig,
            prev_val,
            rng_runner,
        )
        runner_state, _ = jax.lax.scan(_update_step, (runner_state, 0), None, num_updates)
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_mappo_tarmac")
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

        print("Compiling MAPPO-TarMAC SMAX...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
