"""Rollout vs PPO-replay log-prob consistency for mappo_gru actor."""

import jax
import jax.numpy as jnp
import distrax

from talk.environments.mpe.jaxmarl_adapter import build_mpe_env
from talk.experiments.mpe.mappo_gru.mappo_gru import (
    _actor_step,
    _actor_trajectory,
    _mask_team_logits,
)
from talk.networks.gru import ActorDiscreteRNN, ScannedRNN


def main():
    config = {
        "env_name": "MPE_simple_spread_v3",
        "sight_range": -1.0,
        "num_envs": 8,
        "fc_dim_size": 64,
        "gru_hidden_size": 64,
        "activation": "relu",
        "trajectory_scan_unroll": 8,
    }
    adapter = build_mpe_env(
        env_name=config["env_name"],
        env_kwargs={},
        sight_range=config["sight_range"],
    )
    team = [t for t in adapter.team_specs if t.trainable][0]
    action_dims_py = adapter.action_dims_py
    n_team = len(team.agent_indices)
    num_envs = config["num_envs"]
    hidden_size = config["gru_hidden_size"]

    actor = ActorDiscreteRNN(
        action_dim=team.action_dim,
        hidden_size=hidden_size,
        fc_dim_size=config["fc_dim_size"],
        activation=config["activation"],
    )
    rng = jax.random.PRNGKey(0)
    ac_init_h = ScannedRNN.initialize_carry(n_team * num_envs, hidden_size)
    ac_in = (
        jnp.zeros((1, n_team * num_envs, team.obs_dim), dtype=jnp.float32),
        jnp.zeros((1, n_team * num_envs), dtype=bool),
    )
    params = actor.init(rng, ac_init_h, ac_in)
    actor_apply = actor.apply

    obs, _, _, env_state = jax.vmap(adapter.reset)(
        jax.random.split(jax.random.PRNGKey(1), num_envs)
    )
    done = jnp.zeros((num_envs,), dtype=bool)
    h = ac_init_h.reshape(n_team, num_envs, hidden_size)
    T = 16
    obs_hist, done_hist, logp_hist, act_hist = [], [], [], []

    for _ in range(T):
        team_obs = obs[:, team.agent_indices, : team.obs_dim]
        h, pi = _actor_step(
            actor_apply,
            params,
            h,
            team_obs,
            done,
            n_team,
        )
        logits = _mask_team_logits(
            pi.logits.reshape(n_team, num_envs, team.action_dim),
            team,
            action_dims_py,
        )
        masked_pi = distrax.Categorical(logits=logits.transpose(1, 0, 2))
        rng, key = jax.random.split(rng)
        action = masked_pi.sample(seed=key)
        logp = masked_pi.log_prob(action)
        obs_hist.append(team_obs)
        done_hist.append(done)
        logp_hist.append(logp)
        act_hist.append(action)
        rng, keys = jax.random.split(rng)
        actions = jnp.zeros((num_envs, adapter.num_agents), dtype=jnp.int32)
        actions = actions.at[:, team.agent_indices].set(action)
        new_obs, _, _, env_state, _, new_done = jax.vmap(
            lambda k, s, a, ac: adapter.step(k, s, a, ac)[:6]
        )(
            jax.random.split(keys, num_envs),
            env_state,
            actions,
            jnp.zeros(
                (num_envs, adapter.num_agents, adapter.max_action_dim),
                dtype=jnp.float32,
            ),
        )
        obs, done = new_obs, new_done

    init_h = ac_init_h.reshape(n_team, num_envs, hidden_size)
    team_obs_seq = jnp.stack(obs_hist)
    done_seq = jnp.stack(done_hist)
    logp_roll = jnp.stack(logp_hist)
    act_seq = jnp.stack(act_hist)
    logits_replay = _actor_trajectory(
        actor_apply,
        params,
        init_h,
        team_obs_seq,
        done_seq,
        team,
        action_dims_py,
        config["trajectory_scan_unroll"],
    )
    logp_replay = distrax.Categorical(logits=logits_replay).log_prob(act_seq)
    max_diff = float(jnp.max(jnp.abs(logp_roll - logp_replay)))
    print("max |logp_roll - logp_replay|:", max_diff)
    assert max_diff < 1e-5, "replay mismatch"


if __name__ == "__main__":
    main()
