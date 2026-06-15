"""Smoke test: ratio_0 metric exactly as computed in mappo_tarmac training loop."""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from talk.experiments.smax.tarmac.mappo_tarmac import make_train


def main():
    cfg = OmegaConf.to_container(
        OmegaConf.load("talk/experiments/smax/tarmac/config_mappo_tarmac.yaml")
    )
    cfg["num_envs"] = 8
    cfg["num_steps_per_env_per_update"] = 32
    cfg["total_timesteps"] = cfg["num_envs"] * cfg["num_steps_per_env_per_update"]
    cfg["num_seeds"] = 1
    cfg["use_wandb"] = False

    train = make_train(cfg)
    rng = jax.random.PRNGKey(0)

    # Patch train to return ratio diagnostics after one update
    inner_train = train

    def one_update(rng, exp_id):
        # Replicate train() but stop after first update with loss_info
        # Easiest: run full train with 1 update worth of timesteps (already set)
        out = inner_train(rng, exp_id)
        return out

    # Instrument by extracting from a custom version - run make_train internals
    # via executing train with tiny steps and hooking is hard; inline one update:
    from talk.experiments.smax.tarmac import mappo_tarmac as mt
    from talk.networks.gru import ScannedRNN
    from talk.networks.smax import ActorTarMACRNNAvailMasked
    from flax.training.train_state import TrainState
    import optax
    import distrax
    from jaxmarl.wrappers.baselines import SMAXLogWrapper
    from talk.experiments.smax.env_utils import make_smax_env

    num_envs = cfg["num_envs"]
    num_steps = cfg["num_steps_per_env_per_update"]
    num_minibatches = cfg["num_minibatches"]
    num_epochs = cfg["num_epochs"]
    sig_dim = cfg["sig_dim"]
    val_dim = cfg["val_dim"]
    clip_eps = cfg["clip_eps"] / 8  # scale_clip_eps

    env = make_smax_env(cfg, None)
    env = mt.SMAXWorldStateWrapper(env, cfg["obs_with_agent_id"])
    env = SMAXLogWrapper(env)
    num_agents = env.num_agents
    num_actors = num_agents * num_envs
    mb_envs = num_envs // num_minibatches

    # Use closures from make_train by calling it and extracting... 
    # Instead duplicate minimal one-update using make_train's compiled train with 1 update
    result = jax.jit(inner_train)(rng, 0)
    print("Full one-update train completed (sanity).")

    # Now run diagnostic update manually with same config
    actor_network = ActorTarMACRNNAvailMasked(
        action_dim=env.action_space(env.agents[0]).n,
        hidden_size=cfg["gru_hidden_size"],
        fc_dim_size=cfg["fc_dim_size"],
        sig_dim=sig_dim,
        val_dim=val_dim,
        activation=cfg["activation"],
    )
    from talk.networks.gru import CriticDiscreteRNN

    critic_network = CriticDiscreteRNN(
        hidden_size=cfg["gru_hidden_size"],
        fc_dim_size=cfg["fc_dim_size"],
        activation=cfg["activation"],
    )
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    comm_range = float(cfg.get("comm_range", -1))
    scan_unroll = int(cfg.get("trajectory_scan_unroll", 8))

    rng, ra, rc = jax.random.split(rng, 3)
    ac_init_h = ScannedRNN.initialize_carry(num_agents, cfg["gru_hidden_size"])
    actor_params = actor_network.init(
        ra,
        ac_init_h,
        jnp.zeros((num_agents, obs_dim)),
        jnp.zeros((num_agents, sig_dim)),
        jnp.zeros((num_agents, val_dim)),
        jnp.zeros((num_agents,), dtype=bool),
        jnp.ones((num_agents, action_dim)),
        jnp.ones((num_agents, num_agents), dtype=bool),
        method=ActorTarMACRNNAvailMasked.step,
    )
    cr_init_h = ScannedRNN.initialize_carry(num_envs, cfg["gru_hidden_size"])
    critic_params = critic_network.init(
        rc,
        cr_init_h,
        (
            jnp.zeros((1, num_envs, env.world_state_size()), dtype=jnp.float32),
            jnp.zeros((1, num_envs), dtype=bool),
        ),
    )
    tx = optax.adam(1e-3)
    actor_state = TrainState.create(
        apply_fn=actor_network.apply, params=actor_params, tx=tx
    )
    critic_state = TrainState.create(
        apply_fn=critic_network.apply, params=critic_params, tx=tx
    )

    def tarmac_env_step(params, ac_h, obs, avail, done, prev_sig, prev_val, ally_pos):
        h = mt.to_env_major(ac_h, num_envs, num_agents)
        obs_e = mt.to_env_major(obs, num_envs, num_agents)
        avail_e = mt.to_env_major(avail, num_envs, num_agents)
        done_e = mt.to_env_major(done, num_envs, num_agents)

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
            mt.to_actor_major(new_h, num_envs, num_agents),
            sig,
            val,
            mt.to_actor_major(logits, num_envs, num_agents),
        )

    def actor_trajectory(params, init_h, ps0, pv0, traj):
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
            unroll=scan_unroll,
        )
        return distrax.Categorical(logits=logits)

    @jax.jit
    def rollout(params, rng):
        rng, rr = jax.random.split(rng)
        reset_rng = jax.random.split(rr, num_envs)
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        ac_h = ScannedRNN.initialize_carry(num_actors, cfg["gru_hidden_size"])
        prev_sig = jnp.zeros((num_envs, num_agents, sig_dim))
        prev_val = jnp.zeros((num_envs, num_agents, val_dim))
        last_done = jnp.zeros((num_actors,), dtype=bool)
        init_h = ac_h
        ps_init = prev_sig
        pv_init = prev_val

        def env_step(carry, _):
            ac_h, prev_sig, prev_val, env_state, last_obs, last_done, rng = carry
            rng, rng_act = jax.random.split(rng)
            avail = mt.batchify(
                jax.vmap(env.get_avail_actions)(env_state.env_state),
                env.agents,
                num_actors,
            )
            obs_batch = mt.batchify(last_obs, env.agents, num_actors)
            ally_pos = env_state.env_state.state.unit_positions[:, :num_agents, :]
            ac_h, sig, val, logits = tarmac_env_step(
                params, ac_h, obs_batch, avail, last_done, prev_sig, prev_val, ally_pos
            )
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=rng_act)
            log_prob = pi.log_prob(action)
            env_act = mt.unbatchify(action.squeeze(), env.agents, num_envs, num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}
            rng, rs = jax.random.split(rng)
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rs, num_envs), env_state, env_act
            )
            done_batch = mt.batchify(done, env.agents, num_actors).squeeze()
            ep_done = done["__all__"][:, None, None]
            prev_sig_next = jnp.where(ep_done, 0.0, sig)
            prev_val_next = jnp.where(ep_done, 0.0, val)
            tr = mt.Transition(
                global_done=jnp.tile(done["__all__"], env.num_agents),
                done=last_done,
                action=action.squeeze(),
                value=jnp.zeros((num_actors,)),
                reward=mt.batchify(reward, env.agents, num_actors).squeeze(),
                log_prob=log_prob.squeeze(),
                obs=obs_batch,
                world_state=jnp.zeros((num_actors, 1)),
                info={},
                avail_actions=avail,
                ally_positions=mt.to_actor_major(ally_pos, num_envs, num_agents),
            )
            return (ac_h, prev_sig_next, prev_val_next, env_state, obsv, done_batch, rng), tr

        _, traj = jax.lax.scan(
            env_step,
            (ac_h, prev_sig, prev_val, env_state, obsv, last_done, rng),
            None,
            num_steps,
        )
        return init_h, ps_init, pv_init, traj

    @jax.jit
    def compute_ratio_0(params, rng, perm):
        init_h, ps_init, pv_init, traj = rollout(params, rng)
        init_h_env = mt.to_env_major(init_h, num_envs, num_agents)
        traj_env = mt.Transition(
            global_done=mt.traj_field_to_env_major(traj.global_done, num_envs, num_agents),
            done=mt.traj_field_to_env_major(traj.done, num_envs, num_agents),
            action=mt.traj_field_to_env_major(traj.action, num_envs, num_agents),
            value=mt.traj_field_to_env_major(traj.value, num_envs, num_agents),
            reward=mt.traj_field_to_env_major(traj.reward, num_envs, num_agents),
            log_prob=mt.traj_field_to_env_major(traj.log_prob, num_envs, num_agents),
            obs=mt.traj_field_to_env_major(traj.obs, num_envs, num_agents),
            world_state=mt.traj_field_to_env_major(traj.world_state, num_envs, num_agents),
            info={},
            avail_actions=mt.traj_field_to_env_major(traj.avail_actions, num_envs, num_agents),
            ally_positions=mt.traj_field_to_env_major(
                traj.ally_positions, num_envs, num_agents
            ),
        )

        def reshape_mb(x):
            x_shuf = x[:, perm]
            if x.ndim == 3:
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

        traj_mb = mt.Transition(
            global_done=reshape_mb(traj_env.global_done),
            done=reshape_mb(traj_env.done),
            action=reshape_mb(traj_env.action),
            value=reshape_mb(traj_env.value),
            reward=reshape_mb(traj_env.reward),
            log_prob=reshape_mb(traj_env.log_prob),
            obs=reshape_mb(traj_env.obs),
            world_state=reshape_mb(traj_env.world_state),
            info={},
            avail_actions=reshape_mb(traj_env.avail_actions),
            ally_positions=reshape_mb(traj_env.ally_positions),
        )
        init_h_mb = init_h_env[perm].reshape(num_minibatches, mb_envs, num_agents, -1)
        ps_mb = ps_init[perm].reshape(num_minibatches, mb_envs, num_agents, sig_dim)
        pv_mb = pv_init[perm].reshape(num_minibatches, mb_envs, num_agents, val_dim)

        # epoch 0, minibatch 0
        pi = actor_trajectory(
            params, init_h_mb[0], ps_mb[0], pv_mb[0],
            mt.Transition(
                global_done=traj_mb.global_done[0],
                done=traj_mb.done[0],
                action=traj_mb.action[0],
                value=traj_mb.value[0],
                reward=traj_mb.reward[0],
                log_prob=traj_mb.log_prob[0],
                obs=traj_mb.obs[0],
                world_state=traj_mb.world_state[0],
                info={},
                avail_actions=traj_mb.avail_actions[0],
                ally_positions=traj_mb.ally_positions[0],
            ),
        )
        log_prob = pi.log_prob(traj_mb.action[0])
        ratio = jnp.exp(log_prob - traj_mb.log_prob[0])
        ratio_0_slice = ratio  # as in loss before mean
        ratio_0_metric = ratio.at[0, 0, 0].get() if False else ratio  # full slice [0,0] is wrong index
        return ratio, ratio.mean(), jnp.abs(ratio - 1).max()

    rng, rp = jax.random.split(rng)
    perm = jax.random.permutation(rp, num_envs)
    ratio, ratio_mean, max_dev = compute_ratio_0(actor_state.params, rng, perm)
    ratio_np = np.asarray(ratio)

    print("\n=== PPO-style ratio_0 (epoch0, mb0, with env shuffle) ===")
    print(f"ratio mean (mb0): {float(ratio_mean):.6f}")
    print(f"|ratio-1| max: {float(max_dev):.6f}")
    print(f"ratio tensor shape: {ratio_np.shape}")
    # How training indexes ratio_0 after epoch/minibatch scans:
    # loss ratio shape (num_epochs, num_minibatches, T, mb_E, N)
    # ratio.at[0,0].get() -> (T, mb_E, N)
    r00 = ratio_np  # our mb0 only
    print(f"ratio_0 style mean (all t, mb_E, N in mb0): {r00.mean():.6f}")
    print(f"ratio at t=0, env=0, agent=0: {r00[0,0,0]:.6f}")


if __name__ == "__main__":
    main()
