"""Compare rollout vs replay max |ratio-1| for query-before vs query-after reset."""

import numpy as np
import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn
from omegaconf import OmegaConf
from flax.linen.initializers import constant, orthogonal

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
from talk.networks.mlp import _activation_fn
from talk.networks.smax import ActorTarMACRNNAvailMasked, tarmac_aggregate


class QueryBeforeReset(ActorTarMACRNNAvailMasked):
    @nn.compact
    def step(self, hidden, obs, prev_signature, prev_value, done, avail_actions, comm_reachability=None):
        activation = _activation_fn(self.activation)
        alive = ~done.astype(bool)
        query = nn.Dense(
            self.sig_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="query",
        )(hidden)
        comm_ctx = tarmac_aggregate(
            query, prev_signature, prev_value, alive, sender_reachable=comm_reachability
        )
        obs_embed = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        obs_embed = activation(obs_embed)
        gru_in = jnp.concatenate([obs_embed, comm_ctx], axis=-1)
        cell = nn.GRUCell(features=self.hidden_size)
        reset = done.astype(hidden.dtype)
        hidden = jnp.where(reset[:, None], jnp.zeros_like(hidden), hidden)
        new_hidden, _ = cell(hidden, gru_in)
        signature = nn.Dense(
            self.sig_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="signature"
        )(new_hidden)
        value = nn.Dense(
            self.val_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="value"
        )(new_hidden)
        head = activation(
            nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(new_hidden)
        )
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(head)
        invalid = 1.0 - avail_actions.astype(logits.dtype)
        return new_hidden, signature, value, logits - invalid * 1e10


def run_test(net_cls, label):
    cfg = OmegaConf.to_container(
        OmegaConf.load("talk/experiments/smax/tarmac/config_mappo_tarmac.yaml")
    )
    cfg.update(num_envs=8, num_steps_per_env_per_update=32, use_wandb=False)
    env = SMAXLogWrapper(
        mt.SMAXWorldStateWrapper(make_smax_env(cfg, None), cfg["obs_with_agent_id"])
    )
    num_agents = env.num_agents
    num_envs = cfg["num_envs"]
    num_steps = cfg["num_steps_per_env_per_update"]
    num_actors = num_agents * num_envs
    sig_dim, val_dim = cfg["sig_dim"], cfg["val_dim"]
    comm_range = float(cfg.get("comm_range", -1))

    actor_network = net_cls(
        action_dim=env.action_space(env.agents[0]).n,
        hidden_size=cfg["gru_hidden_size"],
        fc_dim_size=cfg["fc_dim_size"],
        sig_dim=sig_dim,
        val_dim=val_dim,
        activation=cfg["activation"],
    )
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    rng = jax.random.PRNGKey(0)
    rng, ra = jax.random.split(rng)
    params = actor_network.init(
        ra,
        ScannedRNN.initialize_carry(num_agents, cfg["gru_hidden_size"]),
        jnp.zeros((num_agents, obs_dim)),
        jnp.zeros((num_agents, sig_dim)),
        jnp.zeros((num_agents, val_dim)),
        jnp.zeros((num_agents,), dtype=bool),
        jnp.ones((num_agents, action_dim)),
        jnp.ones((num_agents, num_agents), dtype=bool),
        method=net_cls.step,
    )

    @jax.jit
    def eval_ratio(rng):
        rng, rr = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(rr, num_envs))
        ac_h = ScannedRNN.initialize_carry(num_actors, cfg["gru_hidden_size"])
        prev_sig = jnp.zeros((num_envs, num_agents, sig_dim))
        prev_val = jnp.zeros((num_envs, num_agents, val_dim))
        last_done = jnp.zeros((num_actors,), dtype=bool)
        init_h_env = to_env_major(ac_h, num_envs, num_agents)

        def step(carry, _):
            ac_h, prev_sig, prev_val, env_state, last_obs, last_done, rng = carry
            rng, rng_act = jax.random.split(rng)
            avail = batchify(
                jax.vmap(env.get_avail_actions)(env_state.env_state), env.agents, num_actors
            )
            obs_batch = batchify(last_obs, env.agents, num_actors)
            ally_pos = env_state.env_state.state.unit_positions[:, :num_agents, :]
            h = to_env_major(ac_h, num_envs, num_agents)
            obs_e = to_env_major(obs_batch, num_envs, num_agents)
            avail_e = to_env_major(avail, num_envs, num_agents)
            done_e = to_env_major(last_done, num_envs, num_agents)

            def one(h_e, o_e, a_e, d_e, ps, pv, pos):
                reach = mt.ally_comm_reachability(pos, comm_range)
                return actor_network.apply(
                    params, h_e, o_e, ps, pv, d_e, a_e, reach, method=net_cls.step
                )

            new_h, sig, val, logits = jax.vmap(one)(
                h, obs_e, avail_e, done_e, prev_sig, prev_val, ally_pos
            )
            ac_h = to_actor_major(new_h, num_envs, num_agents)
            logits = to_actor_major(logits, num_envs, num_agents)
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=rng_act)
            log_prob = pi.log_prob(action)
            env_act = {
                k: v.squeeze()
                for k, v in mt.unbatchify(
                    action.squeeze(), env.agents, num_envs, num_agents
                ).items()
            }
            obsv, env_state, _, done, _ = jax.vmap(env.step)(
                jax.random.split(rng, num_envs), env_state, env_act
            )
            ep_done = done["__all__"][:, None, None]
            tr = Transition(
                global_done=jnp.tile(done["__all__"], num_agents),
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
                jnp.where(ep_done, 0.0, sig),
                jnp.where(ep_done, 0.0, val),
                env_state,
                obsv,
                batchify(done, env.agents, num_actors).squeeze(),
                rng,
            ), tr

        _, traj = jax.lax.scan(
            step,
            (ac_h, prev_sig, prev_val, env_state, obsv, last_done, rng),
            None,
            num_steps,
        )
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
            ally_positions=traj_field_to_env_major(traj.ally_positions, num_envs, num_agents),
        )

        def replay(carry, inp):
            h, ps, pv = carry
            ot, dt, at, pt, gd = inp

            def one(h_e, o_e, d_e, a_e, ps_e, pv_e, pos):
                reach = mt.ally_comm_reachability(pos, comm_range)
                return actor_network.apply(
                    params, h_e, o_e, ps_e, pv_e, d_e, a_e, reach, method=net_cls.step
                )

            nh, s, v, logits = jax.vmap(one)(h, ot, dt, at, ps, pv, pt)
            ep = gd[:, 0:1, None]
            return (nh, jnp.where(ep, 0.0, s), jnp.where(ep, 0.0, v)), logits

        _, logits = jax.lax.scan(
            replay,
            (init_h_env, jnp.zeros((num_envs, num_agents, sig_dim)), jnp.zeros((num_envs, num_agents, val_dim))),
            (
                traj_env.obs,
                traj_env.done,
                traj_env.avail_actions,
                traj_env.ally_positions,
                traj_env.global_done,
            ),
        )
        new_lp = distrax.Categorical(logits=logits).log_prob(traj_env.action)
        ratio = jnp.exp(new_lp - traj_env.log_prob)
        return jnp.abs(ratio - 1).max(), ratio.mean()

    mx, mn = eval_ratio(jax.random.PRNGKey(1))
    print(f"{label}: max|ratio-1|={float(mx):.8f}, mean_ratio={float(mn):.8f}")


if __name__ == "__main__":
    run_test(QueryBeforeReset, "query_before_reset (old)")
    run_test(ActorTarMACRNNAvailMasked, "query_after_reset (new)")
