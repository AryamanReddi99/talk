"""Smoke test: TarMAC rollout log_probs vs PPO replay (_actor_trajectory)."""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import SMAXLogWrapper
from talk.experiments.smax.env_utils import make_smax_env
from talk.experiments.smax.tarmac import mappo_tarmac as mt
from talk.experiments.smax.tarmac.mappo_tarmac import (
    Transition,
    batchify,
    to_actor_major,
    to_env_major,
    traj_field_to_env_major,
)
from talk.networks.gru import ScannedRNN
from talk.networks.smax import ActorTarMACRNNAvailMasked


def main():
    cfg = OmegaConf.to_container(
        OmegaConf.load("talk/experiments/smax/tarmac/config_mappo_tarmac.yaml")
    )
    cfg["num_envs"] = 8
    cfg["num_steps_per_env_per_update"] = 32
    cfg["use_wandb"] = False

    env = make_smax_env(cfg, None)
    env = mt.SMAXWorldStateWrapper(env, cfg["obs_with_agent_id"])
    env = SMAXLogWrapper(env)

    num_agents = env.num_agents
    num_envs = int(cfg["num_envs"])
    num_steps = int(cfg["num_steps_per_env_per_update"])
    num_actors = num_agents * num_envs
    sig_dim = int(cfg["sig_dim"])
    val_dim = int(cfg["val_dim"])
    comm_range = float(cfg.get("comm_range", -1))

    actor_network = ActorTarMACRNNAvailMasked(
        action_dim=env.action_space(env.agents[0]).n,
        hidden_size=cfg["gru_hidden_size"],
        fc_dim_size=cfg["fc_dim_size"],
        sig_dim=sig_dim,
        val_dim=val_dim,
        activation=cfg["activation"],
    )
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    ac_init_h = ScannedRNN.initialize_carry(num_agents, cfg["gru_hidden_size"])
    rng = jax.random.PRNGKey(0)
    rng, rng_actor = jax.random.split(rng)
    params = actor_network.init(
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

    def tarmac_env_step(ac_h, obs, avail, done, prev_sig, prev_val, ally_pos):
        h = to_env_major(ac_h, num_envs, num_agents)
        obs_e = to_env_major(obs, num_envs, num_agents)
        avail_e = to_env_major(avail, num_envs, num_agents)
        done_e = to_env_major(done, num_envs, num_agents)

        def one_env(h_e, o_e, a_e, d_e, ps_e, pv_e, pos_e):
            reach = mt.ally_comm_reachability(pos_e, comm_range)
            return actor_network.apply(
                params,
                h_e,
                o_e,
                ps_e,
                pv_e,
                d_e,
                a_e,
                reach,
                method=ActorTarMACRNNAvailMasked.step,
            )

        new_h, sig, val, logits = jax.vmap(one_env)(
            h, obs_e, avail_e, done_e, prev_sig, prev_val, ally_pos
        )
        return (
            to_actor_major(new_h, num_envs, num_agents),
            sig,
            val,
            to_actor_major(logits, num_envs, num_agents),
        )

    def actor_trajectory(init_h, ps0, pv0, traj):
        def scan_step(carry, inputs):
            h, ps, pv = carry
            obs_t, done_t, avail_t, ally_pos_t, global_done_t = inputs

            def one_env(h_e, o_e, d_e, a_e, ps_e, pv_e, pos_e):
                reach = mt.ally_comm_reachability(pos_e, comm_range)
                return actor_network.apply(
                    params,
                    h_e,
                    o_e,
                    ps_e,
                    pv_e,
                    d_e,
                    a_e,
                    reach,
                    method=ActorTarMACRNNAvailMasked.step,
                )

            new_h, sig, val, logits = jax.vmap(one_env)(
                h, obs_t, done_t, avail_t, ps, pv, ally_pos_t
            )
            ep_done = global_done_t[:, 0:1, None]
            sig = jnp.where(ep_done, 0.0, sig)
            val = jnp.where(ep_done, 0.0, val)
            return (new_h, sig, val), logits

        _, logits = jax.lax.scan(
            scan_step,
            (init_h, ps0, pv0),
            (
                traj.obs,
                traj.done,
                traj.avail_actions,
                traj.ally_positions,
                traj.global_done,
            ),
        )
        return logits

    @jax.jit
    def rollout_and_replay(rng):
        rng, rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(rng_reset, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        ac_h = ScannedRNN.initialize_carry(num_actors, cfg["gru_hidden_size"])
        prev_sig = jnp.zeros((num_envs, num_agents, sig_dim))
        prev_val = jnp.zeros((num_envs, num_agents, val_dim))
        last_done = jnp.zeros((num_actors,), dtype=bool)

        init_h_env = to_env_major(ac_h, num_envs, num_agents)
        ps_init = prev_sig
        pv_init = prev_val

        def env_step(carry, _):
            ac_h, prev_sig, prev_val, env_state, last_obs, last_done, rng = carry
            rng, rng_act = jax.random.split(rng)
            avail = batchify(
                jax.vmap(env.get_avail_actions)(env_state.env_state),
                env.agents,
                num_actors,
            )
            obs_batch = batchify(last_obs, env.agents, num_actors)
            ally_pos = env_state.env_state.state.unit_positions[:, :num_agents, :]
            ac_h, sig, val, logits = tarmac_env_step(
                ac_h, obs_batch, avail, last_done, prev_sig, prev_val, ally_pos
            )
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=rng_act)
            log_prob = pi.log_prob(action)

            env_act = mt.unbatchify(action.squeeze(), env.agents, num_envs, num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}
            rng, rng_step = jax.random.split(rng)
            step_rng = jax.random.split(rng_step, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                step_rng, env_state, env_act
            )
            done_batch = batchify(done, env.agents, num_actors).squeeze()
            global_done = done["__all__"]
            ep_done = global_done[:, None, None]
            prev_sig_next = jnp.where(ep_done, 0.0, sig)
            prev_val_next = jnp.where(ep_done, 0.0, val)

            tr = Transition(
                global_done=jnp.tile(done["__all__"], env.num_agents),
                done=last_done,
                action=action.squeeze(),
                value=jnp.zeros((num_actors,)),
                reward=jnp.zeros((num_actors,)),
                log_prob=log_prob.squeeze(),
                obs=obs_batch,
                world_state=jnp.zeros((num_actors, 1)),
                info={},
                avail_actions=avail,
                ally_positions=to_actor_major(ally_pos, num_envs, num_agents),
            )
            return (
                ac_h,
                prev_sig_next,
                prev_val_next,
                env_state,
                obsv,
                done_batch,
                rng,
            ), tr

        _, traj = jax.lax.scan(env_step, (ac_h, prev_sig, prev_val, env_state, obsv, last_done, rng), None, num_steps)

        traj_env = Transition(
            global_done=traj_field_to_env_major(traj.global_done, num_envs, num_agents),
            done=traj_field_to_env_major(traj.done, num_envs, num_agents),
            action=traj_field_to_env_major(traj.action, num_envs, num_agents),
            value=traj_field_to_env_major(traj.value, num_envs, num_agents),
            reward=traj_field_to_env_major(traj.reward, num_envs, num_agents),
            log_prob=traj_field_to_env_major(traj.log_prob, num_envs, num_agents),
            obs=traj_field_to_env_major(traj.obs, num_envs, num_agents),
            world_state=traj_field_to_env_major(traj.world_state, num_envs, num_agents),
            info={},
            avail_actions=traj_field_to_env_major(traj.avail_actions, num_envs, num_agents),
            ally_positions=traj_field_to_env_major(
                traj.ally_positions, num_envs, num_agents
            ),
        )

        replay_logits = actor_trajectory(init_h_env, ps_init, pv_init, traj_env)
        replay_log_prob = distrax.Categorical(logits=replay_logits).log_prob(traj_env.action)
        ratio = jnp.exp(replay_log_prob - traj_env.log_prob)
        return ratio, traj_env.log_prob, replay_log_prob, traj_env.global_done

    ratio, old_lp, new_lp, global_done = rollout_and_replay(rng)
    ratio_np = np.asarray(ratio)
    diff_lp = np.asarray(new_lp - old_lp)

    print("=== TarMAC rollout vs replay ratio smoke test ===")
    print(f"ratio: min={ratio_np.min():.6f} max={ratio_np.max():.6f} mean={ratio_np.mean():.6f}")
    print(f"|ratio-1|: max={np.abs(ratio_np - 1).max():.6f}")
    print(f"log_prob diff: min={diff_lp.min():.6f} max={diff_lp.max():.6f} mean={diff_lp.mean():.6f}")

    t0_ratio = ratio_np[0]
    print(f"t=0 ratio: min={t0_ratio.min():.6f} max={t0_ratio.max():.6f} mean={t0_ratio.mean():.6f}")

    flat_idx = int(np.argmax(np.abs(diff_lp)))
    t, e, n = np.unravel_index(flat_idx, diff_lp.shape)
    print(f"worst mismatch at t={t}, env={e}, agent={n}: ratio={ratio_np[t,e,n]:.6f}")

    gd = np.asarray(global_done).reshape(num_steps, num_envs, num_agents)
    ep_end_steps = [t for t in range(num_steps) if gd[t, :, 0].any()]
    print(f"steps with episode termination (any env): {ep_end_steps[:10]}")

    if np.abs(ratio_np - 1).max() < 1e-5:
        print("PASS: rollout and replay log_probs match.")
    else:
        print("FAIL: rollout and replay diverge.")
        per_t = np.abs(diff_lp).max(axis=(1, 2))
        for ti in np.argsort(-per_t)[:5]:
            print(f"  t={int(ti)}: max |dlogp|={per_t[ti]:.6f}")


if __name__ == "__main__":
    main()
