"""Message-passing coordination environment for multi-agent RL."""

from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from talk.utils.spaces import Box, Discrete, Space
from talk.utils.typing import PRNGKeyArray, Array, BoolArray, FloatArray, IntArray

NUM_MESSAGE_BITS = 2


def observation_dim(num_agents: int, include_agent_id: bool = False) -> int:
    """Observation size: message bits, optionally plus one-hot agent ID."""
    dim = NUM_MESSAGE_BITS
    if include_agent_id:
        dim += num_agents
    return dim


@struct.dataclass
class State:
    """Internal environment state."""

    ground_truth: Array  # scalar in {0, 1}, fixed for the episode
    step: int
    done: Array


class MessageBox:
    """
    N-agent environment for relaying a hidden binary message.

    Agent 0 observes a fixed binary one-hot ground-truth message for the episode.
    Agent N-1 chooses one of two actions each step; matching the ground truth yields
    +1 reward for agents 0 and N-1, otherwise -1. Intermediate agents (if any) pass
    no-op actions and receive zero reward.
    """

    def __init__(
        self,
        num_agents: int = 2,
        max_steps: int = 10,
        reward_correct: float = 1.0,
        reward_wrong: float = -1.0,
        include_agent_id: bool = False,
    ):
        if num_agents < 2:
            raise ValueError("MessageBox requires at least 2 agents (IDs 0 and N-1).")

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.include_agent_id = include_agent_id
        self.sender_idx = 0
        self.receiver_idx = num_agents - 1

        self.obs_dim = observation_dim(num_agents, include_agent_id)

        self.action_spaces: List[Space] = []
        self.observation_spaces: List[Space] = []
        for i in range(num_agents):
            if i == self.receiver_idx:
                self.action_spaces.append(Discrete(NUM_MESSAGE_BITS))
            else:
                self.action_spaces.append(Discrete(1))
            if include_agent_id:
                self.observation_spaces.append(
                    Box(0.0, 1.0, (self.obs_dim,), dtype=jnp.float32)
                )
            elif i == self.sender_idx:
                self.observation_spaces.append(
                    Box(0.0, 1.0, (NUM_MESSAGE_BITS,), dtype=jnp.float32)
                )
            else:
                self.observation_spaces.append(
                    Box(0.0, 0.0, (NUM_MESSAGE_BITS,), dtype=jnp.float32)
                )

    @property
    def name(self) -> str:
        return "MessageBox"

    def observation_space(self, agent_idx: int) -> Space:
        return self.observation_spaces[agent_idx]

    def action_space(self, agent_idx: int) -> Space:
        return self.action_spaces[agent_idx]

    def _ground_truth_one_hot(self, ground_truth: Array) -> Array:
        return jax.nn.one_hot(ground_truth, NUM_MESSAGE_BITS).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[FloatArray, State]:
        ground_truth = jax.random.randint(
            key, shape=(), minval=0, maxval=NUM_MESSAGE_BITS
        )
        state = State(
            ground_truth=ground_truth,
            step=0,
            done=jnp.array(False),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> FloatArray:
        """Per-agent observations, shape (num_agents, obs_dim)."""
        message = self._ground_truth_one_hot(state.ground_truth)
        task_obs = jnp.zeros((self.num_agents, NUM_MESSAGE_BITS), dtype=jnp.float32)
        task_obs = task_obs.at[self.sender_idx].set(message)
        if self.include_agent_id:
            agent_ids = jnp.eye(self.num_agents, dtype=jnp.float32)
            return jnp.concatenate([task_obs, agent_ids], axis=-1)
        return task_obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: PRNGKeyArray,
        state: State,
        actions: IntArray,
    ) -> Tuple[FloatArray, State, FloatArray, BoolArray, dict]:
        del key  # deterministic transition
        receiver_action = actions[self.receiver_idx]
        correct = receiver_action == state.ground_truth
        step_reward = jnp.where(correct, self.reward_correct, self.reward_wrong).astype(
            jnp.float32
        )

        next_step = state.step + 1
        done = jnp.logical_or(state.done, next_step >= self.max_steps)

        next_state = State(
            ground_truth=state.ground_truth,
            step=next_step,
            done=done,
        )

        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        rewards = rewards.at[self.sender_idx].set(step_reward)
        rewards = rewards.at[self.receiver_idx].set(step_reward)

        return self.get_obs(next_state), next_state, rewards, done, {}

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: State,
        actions: IntArray,
    ) -> Tuple[FloatArray, State, FloatArray, BoolArray, dict]:
        key, key_reset = jax.random.split(key)
        obs_st, state_st, rewards, done, infos = self.step_env(key, state, actions)

        obs_re, state_re = self.reset(key_reset)

        state_out = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs_out = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs_out, state_out, rewards, done, infos
