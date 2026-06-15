"""JaxMARL MPE adapter to batched tensor API used by talk trainers."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

try:
    import jaxmarl  # type: ignore[reportMissingImports]
except ModuleNotFoundError:  # pragma: no cover - local vendored fallback
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(repo_root / "JaxMARL"))
    import jaxmarl  # type: ignore[reportMissingImports]

from talk.environments.mpe.sight_wrapper import LimitedSightWrapper


@dataclass(frozen=True)
class TeamSpec:
    name: str
    agent_names: Tuple[str, ...]
    agent_indices: Tuple[int, ...]
    action_kind: str  # "discrete" | "continuous"
    obs_dim: int
    action_dim: int
    trainable: bool


def _space_kind_and_dim(space) -> Tuple[str, int]:
    if hasattr(space, "n"):
        return "discrete", int(space.n)
    if hasattr(space, "shape"):
        return "continuous", int(space.shape[-1])
    raise ValueError(f"Unsupported space type: {type(space)}")


def _build_team_specs(env, env_name: str) -> List[TeamSpec]:
    classes = env.agent_classes()
    name_to_idx = {a: i for i, a in enumerate(env.agents)}
    specs: List[TeamSpec] = []

    # Merge leader/adversary into one adversary team for world_comm-style envs.
    merged_classes = {}
    for cls_name, members in classes.items():
        mapped = (
            "adversaries" if cls_name in {"leadadversary", "adversaries"} else cls_name
        )
        if isinstance(members, str):
            members = [members]
        merged_classes.setdefault(mapped, [])
        merged_classes[mapped].extend(members)

    for cls_name, members in merged_classes.items():
        if not members:
            continue
        obs_dims = [int(env.observation_space(agent).shape[-1]) for agent in members]
        kinds, act_dims = zip(
            *[_space_kind_and_dim(env.action_space(agent)) for agent in members]
        )
        if len(set(kinds)) != 1:
            raise ValueError(f"Mixed action kinds in team {cls_name}: {kinds}")
        kind = kinds[0]
        trainable = not (
            env_name.startswith("MPE_simple_facmac") and cls_name == "agents"
        )
        specs.append(
            TeamSpec(
                name=cls_name,
                agent_names=tuple(members),
                agent_indices=tuple(name_to_idx[a] for a in members),
                action_kind=kind,
                obs_dim=max(obs_dims),
                action_dim=max(act_dims),
                trainable=trainable,
            )
        )

    # Keep deterministic ordering: adversaries first, then agents.
    specs.sort(key=lambda s: (0 if "adversar" in s.name else 1, s.name))
    return specs


class MPEAdapter:
    """Dict-based MPE API -> padded array API."""

    def __init__(self, env, env_name: str):
        self.env = env
        self.env_name = env_name
        self.agents = list(env.agents)
        self.num_agents = len(self.agents)
        self.agent_to_idx = {a: i for i, a in enumerate(self.agents)}
        self.team_specs = _build_team_specs(env, env_name)

        self.obs_dims = jnp.array(
            [int(env.observation_space(a).shape[-1]) for a in self.agents],
            dtype=jnp.int32,
        )
        self.obs_dims_py = [
            int(env.observation_space(a).shape[-1]) for a in self.agents
        ]
        self.max_obs_dim = int(self.obs_dims.max())
        self.obs_mask = (
            jnp.arange(self.max_obs_dim)[None, :] < self.obs_dims[:, None]
        ).astype(jnp.float32)

        action_dims = []
        self.action_kinds = []
        for a in self.agents:
            kind, dim = _space_kind_and_dim(env.action_space(a))
            self.action_kinds.append(kind)
            action_dims.append(dim)
        self.action_dims = jnp.array(action_dims, dtype=jnp.int32)
        self.action_dims_py = action_dims
        self.max_action_dim = int(self.action_dims.max())
        self.trainable_agent_indices = jnp.array(
            sorted(
                {
                    idx
                    for team in self.team_specs
                    if team.trainable
                    for idx in team.agent_indices
                }
            ),
            dtype=jnp.int32,
        )

    def _dict_obs_to_array(self, obs_dict: Dict[str, chex.Array]) -> chex.Array:
        out = jnp.zeros((self.num_agents, self.max_obs_dim), dtype=jnp.float32)
        for a in self.agents:
            i = self.agent_to_idx[a]
            obs = obs_dict[a].astype(jnp.float32)
            out = out.at[i, : self.obs_dims_py[i]].set(obs)
        return out

    def _dict_reward_to_array(self, reward_dict: Dict[str, chex.Array]) -> chex.Array:
        return jnp.stack([reward_dict[a] for a in self.agents], axis=0).astype(
            jnp.float32
        )

    def reset(self, key: chex.PRNGKey):
        obs_dict, state = self.env.reset(key)
        obs = self._dict_obs_to_array(obs_dict)
        unmasked, positions = self._extras_from_state(state)
        return obs, unmasked, positions, state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        actions_discrete: chex.Array,
        actions_cont: chex.Array,
    ):
        actions = self.actions_to_dict(actions_discrete, actions_cont)
        obs_dict, new_state, reward_dict, done_dict, info = self.env.step(
            key, state, actions
        )
        obs = self._dict_obs_to_array(obs_dict)
        unmasked, positions = self._extras_from_state(new_state)
        rewards = self._dict_reward_to_array(reward_dict)
        done = done_dict["__all__"]
        return obs, unmasked, positions, new_state, rewards, done, info

    def actions_to_dict(
        self, actions_discrete: chex.Array, actions_cont: chex.Array
    ) -> Dict[str, chex.Array]:
        out = {}
        for i, a in enumerate(self.agents):
            if self.action_kinds[i] == "discrete":
                out[a] = actions_discrete[i].astype(jnp.int32)
            else:
                out[a] = actions_cont[i, : self.action_dims_py[i]].astype(jnp.float32)
        return out

    def _base_env(self):
        env = self.env
        while isinstance(env, LimitedSightWrapper):
            env = env._env
        return env

    def _extras_from_state(self, state) -> Tuple[chex.Array, chex.Array]:
        unmasked = self.unmasked_obs_from_state(state)
        positions = self.agent_positions_from_state(state)
        return unmasked, positions

    def unmasked_obs_from_state(self, state) -> chex.Array:
        """Per-agent observations with sight_range=-1 (full relative info)."""
        obs_dict = self._base_env().get_obs(state)
        return self._dict_obs_to_array(obs_dict)

    def agent_positions_from_state(self, state) -> chex.Array:
        """Agent positions with shape (..., num_agents, pos_dim)."""
        return state.p_pos[..., : self.num_agents, :]


def critic_state_dim(num_agents: int, max_obs_dim: int) -> int:
    """Joint concat of all agents' obs plus one-hot global agent id."""
    return int(num_agents * max_obs_dim + num_agents)


def team_critic_state_from_unmasked(
    unmasked: chex.Array,
    team_indices: chex.Array,
    num_agents: int,
    max_obs_dim: int,
) -> chex.Array:
    """
    Critic input per team agent: concat all agents' unmasked obs + one-hot
    global agent id (..., n_team, num_agents * max_obs_dim + num_agents).
    """
    joint = unmasked.reshape(*unmasked.shape[:-2], num_agents * max_obs_dim)
    n_team = team_indices.shape[0]
    agent_onehot = jnp.eye(num_agents, dtype=unmasked.dtype)[team_indices]
    prefix = joint.shape[:-1]
    joint_b = jnp.broadcast_to(
        joint[..., None, :], prefix + (n_team, joint.shape[-1])
    )
    ids_b = jnp.broadcast_to(agent_onehot, prefix + (n_team, num_agents))
    return jnp.concatenate([joint_b, ids_b], axis=-1)


def build_mpe_env(
    env_name: str,
    env_kwargs: Dict,
    sight_range: float,
    num_agents: Optional[int] = None,
) -> MPEAdapter:
    if num_agents is not None:
        env_kwargs["num_agents"] = int(num_agents)
        if env_name == "MPE_simple_spread_v3" and "num_landmarks" not in env_kwargs:
            env_kwargs["num_landmarks"] = int(num_agents)
    # Keep MPE action spaces discrete by default (including world_comm); FACMAC is continuous.
    env_kwargs.setdefault("action_type", "Discrete")
    env = jaxmarl.make(env_name, **env_kwargs)
    if sight_range >= 0:
        env = LimitedSightWrapper(env, env_name=env_name, sight_range=sight_range)
    return MPEAdapter(env, env_name=env_name)
