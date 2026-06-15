"""Team-shared MAPPO + communication on JaxMARL MPE."""

import datetime
from typing import Any, Dict, NamedTuple, Optional

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from talk.environments.mpe.rollout_viz import log_mappo_talk_rollout_video_callback
from talk.experiments.mpe.env_utils import (
    ally_comm_reachability,
    batchify,
    make_mpe_train_env,
    mpe_agent_positions,
    to_actor_major,
    to_env_major,
    traj_field_to_env_major,
    unbatchify,
)
from talk.networks.gru import CriticDiscreteRNN, ScannedRNN
from talk.networks.talk_decoder import ActorTalkRNN
from talk.utils.wandb_multilogger import WandbMultiLogger

LOGGER: Optional[WandbMultiLogger] = None


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
    ally_positions: jnp.ndarray
    msg_key: jnp.ndarray


def make_train(config: Dict[str, Any]):
    env = make_mpe_train_env(config)

    num_agents = env.num_agents
    num_envs = int(config["num_envs"])
    num_steps = int(config["num_steps_per_env_per_update"])
    num_actors = num_agents * num_envs
    num_updates = int(config["total_timesteps"]) // num_steps // num_envs
    config["num_actors"] = num_actors
    config["num_updates"] = num_updates

    vocab_content = int(config["vocab_content"])
    vocab_embed_dim = int(config["vocab_embed_dim"])
    attn_dim = int(config["attn_dim"])
    max_msg_len = int(config["max_msg_len"])
    gumbel_tau = float(config["gumbel_tau"])
    len_aux_coef = float(config["len_aux_coef"])
    vocab_size = vocab_content + 2
    comm_range = float(config.get("comm_range", -1))
    scan_unroll = int(config.get("trajectory_scan_unroll", 8))
    num_minibatches = int(config["num_minibatches"])
    assert num_envs % num_minibatches == 0, "num_envs must divide num_minibatches"
    mb_envs = num_envs // num_minibatches

    clip_eps = float(config["clip_eps"])
    if config.get("scale_clip_eps", False):
        clip_eps = clip_eps / num_agents

    actor_lr = float(config["lr_actor"])
    critic_lr = float(config["lr_critic"])

    def _actor_lr_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["num_epochs"])) / num_updates
        return actor_lr * frac

    def _critic_lr_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["num_epochs"])) / num_updates
        return critic_lr * frac

    log_rollout_videos = config.get("log_rollout_videos", True) and config.get("use_wandb", True)
    rollout_fractions = config.get("log_rollout_fractions", [0.0, 0.25, 0.5, 0.75, 1.0])
    checkpoint_steps = jnp.array(
        sorted(
            {
                int(round(float(fraction) * max(num_updates - 1, 0)))
                for fraction in rollout_fractions
            }
        ),
        dtype=jnp.int32,
    )
    rollout_ctx = {
        "env_name": config["env_name"],
        "env_kwargs": dict(config.get("env_kwargs", {}) or {}),
        "sight_range": config["sight_range"],
        "comm_range": comm_range,
        "num_agents": config.get("num_agents"),
        "activation": config["activation"],
        "fc_dim_size": config["fc_dim_size"],
        "gru_hidden_size": config["gru_hidden_size"],
        "vocab_content": vocab_content,
        "vocab_embed_dim": vocab_embed_dim,
        "attn_dim": attn_dim,
        "max_msg_len": max_msg_len,
        "gumbel_tau": gumbel_tau,
        "rollout_length_multiplier": float(config.get("rollout_length_multiplier", 1.0)),
        "rollout_eval_seed": int(config.get("rollout_eval_seed", 42)),
        "num_update_steps": num_updates,
        "log_seed": 0,
    }

    def rollout_io_callback(should_log, exp_id, update_step, log_step, actor_params):
        log_mappo_talk_rollout_video_callback(
            should_log,
            exp_id,
            update_step,
            actor_params,
            rollout_ctx,
            LOGGER,
            log_step=log_step,
        )

    def _talk_env_step(
        actor_apply,
        params,
        ac_h_flat: jnp.ndarray,
        obs_flat: jnp.ndarray,
        done_flat: jnp.ndarray,
        prev_tokens: jnp.ndarray,
        prev_valid: jnp.ndarray,
        msg_keys_flat: jnp.ndarray,
        ally_pos: jnp.ndarray,
    ):
        h = to_env_major(ac_h_flat, num_envs, num_agents)
        obs = to_env_major(obs_flat, num_envs, num_agents)
        done = to_env_major(done_flat, num_envs, num_agents)
        msg_keys = to_env_major(msg_keys_flat, num_envs, num_agents)

        def _one_env(h_e, obs_e, done_e, pt_e, pv_e, key_e, pos_e):
            reach = ally_comm_reachability(pos_e, comm_range)
            return actor_apply(
                params,
                h_e,
                obs_e,
                pt_e,
                pv_e,
                done_e,
                reach,
                key_e,
                method=ActorTalkRNN.step,
            )

        new_h, logits, msg_tokens, msg_valid, _, _ = jax.vmap(_one_env)(
            h, obs, done, prev_tokens, prev_valid, msg_keys, ally_pos
        )
        return (
            to_actor_major(new_h, num_envs, num_agents),
            msg_tokens,
            msg_valid,
            to_actor_major(logits, num_envs, num_agents),
        )

    def _actor_trajectory(
        actor_apply,
        params,
        init_h: jnp.ndarray,
        init_prev_tokens: jnp.ndarray,
        init_prev_valid: jnp.ndarray,
        traj,
    ):
        def _scan_step(carry, inputs):
            h, pt, pv = carry
            obs_t, done_t, ally_pos_t, global_done_t, msg_key_t = inputs

            def _one_env(h_e, obs_e, done_e, pt_e, pv_e, key_e, pos_e):
                reach = ally_comm_reachability(pos_e, comm_range)
                return actor_apply(
                    params,
                    h_e,
                    obs_e,
                    pt_e,
                    pv_e,
                    done_e,
                    reach,
                    key_e,
                    method=ActorTalkRNN.step,
                )

            new_h, logits, msg_tokens, msg_valid, expected_len, comm_ctx = jax.vmap(
                _one_env
            )(h, obs_t, done_t, pt, pv, msg_key_t, ally_pos_t)
            ep_done = global_done_t[:, :, None]
            msg_tokens = jnp.where(ep_done[..., None], 0.0, msg_tokens)
            msg_valid = jnp.where(ep_done, False, msg_valid)
            return (new_h, msg_tokens, msg_valid), (
                logits,
                expected_len,
                msg_tokens,
                msg_valid,
                comm_ctx,
            )

        _, (logits, expected_len, msg_tokens, msg_valid, comm_ctx) = jax.lax.scan(
            _scan_step,
            (init_h, init_prev_tokens, init_prev_valid),
            (
                traj.obs,
                traj.done,
                traj.ally_positions,
                traj.global_done,
                traj.msg_key,
            ),
            unroll=scan_unroll,
        )
        return (
            distrax.Categorical(logits=logits),
            expected_len,
            msg_tokens,
            msg_valid,
            comm_ctx,
        )

    def train(rng: jnp.ndarray, exp_id: int):
        obs_dim = int(env.observation_space(env.agents[0]).shape[0])
        action_dim = int(env.action_space(env.agents[0]).n)
        world_state_dim = env.world_state_size()

        actor_network = ActorTalkRNN(
            action_dim=action_dim,
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            vocab_content=vocab_content,
            vocab_embed_dim=vocab_embed_dim,
            attn_dim=attn_dim,
            max_msg_len=max_msg_len,
            gumbel_tau=gumbel_tau,
            activation=config["activation"],
        )
        critic_network = CriticDiscreteRNN(
            hidden_size=config["gru_hidden_size"],
            fc_dim_size=config["fc_dim_size"],
            activation=config["activation"],
        )

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        ac_init_h = ScannedRNN.initialize_carry(num_agents, config["gru_hidden_size"])
        actor_params = actor_network.init(
            rng_actor,
            ac_init_h,
            jnp.zeros((num_agents, obs_dim)),
            jnp.zeros((num_agents, max_msg_len, vocab_size)),
            jnp.zeros((num_agents, max_msg_len), dtype=bool),
            jnp.zeros((num_agents,), dtype=bool),
            jnp.ones((num_agents, num_agents), dtype=bool),
            jax.random.split(rng_actor, num_agents),
            method=ActorTalkRNN.step,
        )

        cr_init_x = (
            jnp.zeros((1, num_actors, world_state_dim), dtype=jnp.float32),
            jnp.zeros((1, num_actors), dtype=bool),
        )
        cr_init_h = ScannedRNN.initialize_carry(num_actors, config["gru_hidden_size"])
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
        prev_tokens = jnp.zeros((num_envs, num_agents, max_msg_len, vocab_size))
        prev_valid = jnp.zeros((num_envs, num_agents, max_msg_len), dtype=bool)

        def _update_step(update_runner_state, _):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, _):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    hstates,
                    prev_tokens_carry,
                    prev_valid_carry,
                    rng,
                ) = runner_state

                rng, rng_act, rng_msg = jax.random.split(rng, 3)
                obs_batch = batchify(last_obs, env.agents, num_actors)
                ally_pos = mpe_agent_positions(env_state, num_agents)
                msg_keys = jax.random.split(rng_msg, num_actors)
                ac_h, msg_tokens, msg_valid, logits = _talk_env_step(
                    actor_network.apply,
                    train_states[0].params,
                    hstates[0],
                    obs_batch,
                    last_done,
                    prev_tokens_carry,
                    prev_valid_carry,
                    msg_keys,
                    ally_pos,
                )
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action.squeeze(), env.agents, num_envs, num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                world_state = last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))
                cr_in = (world_state[None, :], last_done[None, :])
                cr_h, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                rng, rng_step = jax.random.split(rng)
                step_rng = jax.random.split(rng_step, num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    step_rng, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                done_batch = batchify(done, env.agents, num_actors).squeeze()
                global_done = done["__all__"]
                ep_done = global_done[:, None, None]
                prev_tokens_next = jnp.where(ep_done[..., None], 0.0, msg_tokens)
                prev_valid_next = jnp.where(ep_done, False, msg_valid)

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
                    ally_positions=to_actor_major(ally_pos, num_envs, num_agents),
                    msg_key=msg_keys,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_h, cr_h),
                    prev_tokens_next,
                    prev_valid_next,
                    rng,
                )
                return runner_state, transition

            init_hstates = runner_state[4]
            prev_tokens_init = runner_state[5]
            prev_valid_init = runner_state[6]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, num_steps)
            (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                prev_tokens,
                prev_valid,
                rng,
            ) = runner_state

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
                    *x_env.shape[3:],
                ).swapaxes(1, 0)

            def _update_epoch(update_state, _):
                def _update_minbatch(train_states, batch_info):
                    actor_state, critic_state = train_states
                    ac_h, cr_h, pt_init, pv_init, traj_mb, gae_mb, target_mb = batch_info

                    def _actor_loss(params, init_h, pt0, pv0, mb_traj, mb_gae):
                        pi, expected_len, msg_tokens, msg_valid, comm_ctx = (
                            _actor_trajectory(
                                actor_network.apply,
                                params,
                                init_h,
                                pt0,
                                pv0,
                                mb_traj,
                            )
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
                        expected_msg_len = expected_len.mean()
                        actor_loss = (
                            loss_actor
                            - config["ent_coef"] * entropy
                            + len_aux_coef * expected_msg_len
                        )

                        valid_f = msg_valid.astype(jnp.float32)
                        lengths = valid_f.sum(axis=-1)
                        true_msg_len = lengths.mean()
                        silence_rate = (lengths == 0).mean()
                        comm_ctx_norm = jnp.linalg.norm(comm_ctx, axis=-1).mean()

                        token_counts = (
                            msg_tokens * valid_f[..., None]
                        ).sum(axis=(0, 1, 2, 3))
                        content_counts = token_counts[2:]
                        token_probs = content_counts / (content_counts.sum() + 1e-8)
                        token_entropy = -jnp.sum(
                            token_probs * jnp.log(token_probs + 1e-12)
                        )

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            expected_msg_len,
                            true_msg_len,
                            silence_rate,
                            token_entropy,
                            comm_ctx_norm,
                        )

                    def _critic_loss(params, init_h, mb_traj, mb_targets):
                        mb_actors = mb_envs * num_agents
                        ws = mb_traj.world_state.reshape(
                            mb_traj.world_state.shape[0], mb_actors, -1
                        )
                        done = mb_traj.done.reshape(mb_traj.done.shape[0], mb_actors)
                        init_h_flat = init_h.reshape(mb_actors, -1)
                        _, value = critic_network.apply(params, init_h_flat, (ws, done))
                        value = value.reshape(mb_traj.value.shape)
                        clipped = mb_traj.value + (value - mb_traj.value).clip(-clip_eps, clip_eps)
                        loss_v = jnp.square(value - mb_targets)
                        loss_v_clipped = jnp.square(clipped - mb_targets)
                        value_loss = 0.5 * jnp.maximum(loss_v, loss_v_clipped).mean()
                        critic_loss = config["vf_coef"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    actor_out, actor_grads = actor_grad_fn(
                        actor_state.params, ac_h, pt_init, pv_init, traj_mb, gae_mb
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
                        "expected_msg_len": actor_out[1][5],
                        "msg_len": actor_out[1][6],
                        "silence_rate": actor_out[1][7],
                        "token_entropy": actor_out[1][8],
                        "comm_ctx_norm": actor_out[1][9],
                        "actor_grad_norm": optax.global_norm(actor_grads),
                        "critic_grad_norm": optax.global_norm(critic_grads),
                        "gumbel_tau": jnp.asarray(gumbel_tau, dtype=jnp.float32),
                    }
                    return (actor_state, critic_state), loss_info

                (
                    train_states,
                    init_hstates,
                    prev_tokens_init,
                    prev_valid_init,
                    trajectory,
                    gae,
                    target,
                    rng,
                ) = update_state
                rng, rng_perm = jax.random.split(rng)

                init_h_env = to_env_major(init_hstates[0], num_envs, num_agents)
                init_c_env = to_env_major(init_hstates[1], num_envs, num_agents)
                prev_tokens_env = prev_tokens_init
                prev_valid_env = prev_valid_init
                traj_env = Transition(
                    global_done=traj_field_to_env_major(
                        trajectory.global_done, num_envs, num_agents
                    ),
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
                    ally_positions=traj_field_to_env_major(
                        trajectory.ally_positions, num_envs, num_agents
                    ),
                    msg_key=traj_field_to_env_major(
                        trajectory.msg_key, num_envs, num_agents
                    ),
                )
                gae_env = traj_field_to_env_major(gae, num_envs, num_agents)
                target_env = traj_field_to_env_major(target, num_envs, num_agents)

                perm = jax.random.permutation(rng_perm, num_envs)
                init_h_mb = init_h_env[perm].reshape(num_minibatches, mb_envs, num_agents, -1)
                init_c_mb = init_c_env[perm].reshape(num_minibatches, mb_envs, num_agents, -1)
                prev_tokens_mb = prev_tokens_env[perm].reshape(
                    num_minibatches, mb_envs, num_agents, max_msg_len, vocab_size
                )
                prev_valid_mb = prev_valid_env[perm].reshape(
                    num_minibatches, mb_envs, num_agents, max_msg_len
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
                    info={},
                    ally_positions=_reshape_env_minibatches(traj_env.ally_positions, perm),
                    msg_key=_reshape_env_minibatches(traj_env.msg_key, perm),
                )
                gae_mb = _reshape_env_minibatches(gae_env, perm)
                target_mb = _reshape_env_minibatches(target_env, perm)

                minibatches = (
                    init_h_mb,
                    init_c_mb,
                    prev_tokens_mb,
                    prev_valid_mb,
                    traj_mb,
                    gae_mb,
                    target_mb,
                )
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    init_hstates,
                    prev_tokens_init,
                    prev_valid_init,
                    trajectory,
                    gae,
                    target,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                init_hstates,
                prev_tokens_init,
                prev_valid_init,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["num_epochs"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = jax.tree.map(
                lambda x: x.reshape((num_steps, num_envs, num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            def _log_callback(metric, exp_id):
                episode_mask = metric["returned_episode"][:, :, 0]
                returns = metric["returned_episode_returns"][:, :, 0][episode_mask].mean()
                env_step = metric["update_steps"] * num_envs * num_steps
                payload = {
                    "returns": returns,
                    "env_step": env_step,
                    **metric["loss"],
                }
                if LOGGER is not None:
                    log_payload = {k: float(np.asarray(v)) for k, v in payload.items()}
                    LOGGER.log(
                        int(exp_id),
                        log_payload,
                        step=int(np.asarray(env_step)),
                    )

            metric["update_steps"] = update_step
            if config["use_wandb"]:
                jax.experimental.io_callback(_log_callback, None, metric, exp_id)

            if log_rollout_videos:
                env_step = update_step * num_envs * num_steps
                log_now = jnp.logical_and(
                    exp_id == 0,
                    jnp.any(update_step == checkpoint_steps),
                )
                jax.experimental.io_callback(
                    rollout_io_callback,
                    None,
                    log_now,
                    exp_id,
                    update_step,
                    env_step,
                    train_states[0].params,
                )

            update_step = update_step + 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                prev_tokens,
                prev_valid,
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
            prev_tokens,
            prev_valid,
            rng_runner,
        )
        runner_state, _ = jax.lax.scan(_update_step, (runner_state, 0), None, num_updates)
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path=".", config_name="config_mappo_talk")
def main(config):
    global LOGGER
    try:
        config = OmegaConf.to_container(config)
        group = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config["use_wandb"]:
            LOGGER = WandbMultiLogger(
                project=config["project"],
                group=group,
                job_type=config["algorithm"] + config.get("custom_name", ""),
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

        print("Compiling MAPPO-TarMAC MPE...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
