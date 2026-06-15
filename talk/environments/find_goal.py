"""FindGoal multi-agent grid world environment (JAX port of marl-ae-comm)."""

from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from talk.utils.spaces import Box, Discrete, Space
from talk.utils.typing import PRNGKeyArray, Array, BoolArray, FloatArray, IntArray

# Grid cell types
CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2

# Movement actions (also set facing direction)
ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_STAY = 4
NUM_ACTIONS = 5

# Direction vectors: right, down, left, up
DIR_VECS = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)

# RGB palette (uint8), matching marl-ae-comm objects.py
_PALETTE = (
    jnp.array(
        [
            [35, 25, 30],  # shadow / out-of-bounds
            [100, 100, 100],  # wall grey
            [0, 255, 0],  # goal green
            [255, 0, 0],  # agent red
            [0, 0, 255],  # agent blue
            [112, 39, 195],  # agent purple
            [255, 165, 0],  # agent orange
            [128, 128, 0],  # agent olive
            [255, 0, 189],  # agent pink
        ],
        dtype=jnp.float32,
    )
    / 255.0
)

_SHADOW = 0
_WALL_COLOR = 1
_GOAL_COLOR = 2
_AGENT_COLOR_OFFSET = 3

MAX_PLACEMENT_TRIES = 100


def _make_border_grid(width: int, height: int) -> Array:
    """Grid with wall border and empty interior."""
    grid = jnp.full((width, height), CELL_WALL, dtype=jnp.int32)
    if width > 2 and height > 2:
        grid = grid.at[1 : width - 1, 1 : height - 1].set(CELL_EMPTY)
    return grid


def _interior_coords(width: int, height: int) -> Array:
    xs = jnp.arange(1, width - 1, dtype=jnp.int32)
    ys = jnp.arange(1, height - 1, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def _agent_color_index(agent_idx: Array) -> Array:
    return _AGENT_COLOR_OFFSET + (agent_idx % 6)


@struct.dataclass
class State:
    """Internal environment state."""

    grid: Array  # (W, H) int32: empty/wall/goal
    goal_pos: Array  # (2,)
    agent_pos: Array  # (N, 2)
    agent_dir: IntArray  # (N,)
    agent_done: BoolArray  # (N,)
    agent_active: BoolArray  # (N,)
    adv_mask: BoolArray  # (N,) True for adversaries
    step: int
    done: Array  # scalar bool


def _cell_color(
    grid: Array,
    agent_pos: Array,
    agent_active: BoolArray,
    x: Array,
    y: Array,
) -> Array:
    """Return palette index for world cell (x, y)."""
    width, height = grid.shape
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = jnp.clip(x, 0, width - 1)
    safe_y = jnp.clip(y, 0, height - 1)
    cell = grid[safe_x, safe_y]

    on_goal = in_bounds & (cell == CELL_GOAL)
    on_wall = in_bounds & (cell == CELL_WALL)

    agents_here = agent_active & (agent_pos[:, 0] == x) & (agent_pos[:, 1] == y)
    top_agent = jnp.argmax(agents_here.astype(jnp.int32))
    has_agent = jnp.any(agents_here)

    color = jnp.where(
        ~in_bounds,
        _SHADOW,
        jnp.where(
            on_wall,
            _WALL_COLOR,
            jnp.where(
                has_agent,
                _agent_color_index(top_agent),
                jnp.where(on_goal, _GOAL_COLOR, _SHADOW),
            ),
        ),
    )
    return color


def _render_agent_pov(
    grid: Array,
    agent_pos: Array,
    agent_active: BoolArray,
    agent_idx: int,
    view_size: int,
    view_tile_size: int,
) -> FloatArray:
    """Render POV image for one agent, shape (view_px, view_px, 3)."""
    width, height = grid.shape
    view_px = view_size * view_tile_size
    hs = view_size // 2
    ax, ay = agent_pos[agent_idx, 0], agent_pos[agent_idx, 1]
    top_x = ax - hs
    top_y = ay - hs

    vi = jnp.arange(view_size, dtype=jnp.int32) + top_x
    vj = jnp.arange(view_size, dtype=jnp.int32) + top_y
    wx, wy = jnp.meshgrid(vi, vj, indexing="ij")

    def color_at(x, y):
        return _cell_color(grid, agent_pos, agent_active, x, y)

    color_idx = jax.vmap(jax.vmap(color_at))(wx, wy)

    tile_colors = _PALETTE[color_idx]  # (view_size, view_size, 3)
    pixels = jnp.repeat(
        jnp.repeat(tile_colors, view_tile_size, axis=0),
        view_tile_size,
        axis=1,
    )

    inactive = ~agent_active[agent_idx]
    shadow_img = jnp.broadcast_to(_PALETTE[_SHADOW], (view_px, view_px, 3))
    return jnp.where(inactive, shadow_img, pixels)


def _can_move_to(
    grid: Array,
    agent_pos: Array,
    agent_active: BoolArray,
    agent_idx: int,
    fwd_pos: Array,
    num_agents: int,
) -> BoolArray:
    x, y = fwd_pos[0], fwd_pos[1]
    width, height = grid.shape
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    cell = grid[jnp.clip(x, 0, width - 1), jnp.clip(y, 0, height - 1)]
    not_wall = cell != CELL_WALL
    is_goal = cell == CELL_GOAL

    agent_ids = jnp.arange(num_agents)
    others_here = jnp.any(
        (agent_ids != agent_idx)
        & agent_active
        & (agent_pos[:, 0] == x)
        & (agent_pos[:, 1] == y)
    )
    return in_bounds & not_wall & (is_goal | ~others_here)


def _finalize_rewards(
    step_rewards: FloatArray,
    agent_done: BoolArray,
    adv_mask: BoolArray,
    team_reward_type: str,
    team_reward_multiplier: float,
) -> FloatArray:
    """Apply FindGoalMultiGrid.update_reward post-processing."""
    non_adv_mask = ~adv_mask
    nonadv_done = jnp.all(jnp.where(non_adv_mask, agent_done, True))

    if team_reward_type == "const":
        team_rwd = jnp.where(nonadv_done, team_reward_multiplier, 0.0).astype(
            jnp.float32
        )
        step_rewards = step_rewards + jnp.where(non_adv_mask, team_rwd, 0.0)

    num_adv = jnp.sum(adv_mask.astype(jnp.int32))
    adv_rew = -jnp.sum(jnp.where(non_adv_mask, step_rewards, 0.0))
    adv_share = jnp.where(num_adv > 0, adv_rew / num_adv.astype(jnp.float32), 0.0)
    step_rewards = step_rewards + jnp.where(adv_mask, adv_share, 0.0)
    return step_rewards


class FindGoal:
    """
    N-agent grid world: reach a shared green goal as fast as possible.

    Port of marl-ae-comm FindGoalMultiGrid with MessageBox-compatible JAX API.
    """

    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 15,
        max_steps: int = 512,
        view_size: int = 7,
        view_tile_size: int = 6,
        clutter_density: float = 0.15,
        n_clutter: Optional[int] = None,
        randomize_goal: bool = True,
        can_overlap: bool = False,
        see_through_walls: bool = True,
        observe_self_position: bool = True,
        team_reward_type: str = "share",
        team_reward_multiplier: float = 1.0,
        num_adversaries: int = 0,
        active_after_done: bool = False,
        goal_reward: float = 1.0,
    ):
        del can_overlap, see_through_walls  # encoded in movement rules / POV logic

        if num_agents < 1:
            raise ValueError("FindGoal requires at least 1 agent.")
        if num_adversaries > num_agents:
            raise ValueError("num_adversaries cannot exceed num_agents.")
        if team_reward_type not in ("share", "const"):
            raise ValueError("team_reward_type must be 'share' or 'const'.")

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.view_size = view_size
        self.view_tile_size = view_tile_size
        self.view_px = view_size * view_tile_size
        self.randomize_goal = randomize_goal
        self.observe_self_position = observe_self_position
        self.team_reward_type = team_reward_type
        self.team_reward_multiplier = team_reward_multiplier
        self.num_adversaries = num_adversaries
        self.active_after_done = active_after_done
        self.goal_reward = goal_reward

        if n_clutter is not None:
            self.n_clutter = n_clutter
        else:
            interior = (grid_size - 2) * (grid_size - 2)
            self.n_clutter = int(clutter_density * interior)

        needed = 1 + self.n_clutter + num_agents
        interior = (grid_size - 2) * (grid_size - 2)
        if needed > interior:
            raise ValueError(
                f"Not enough interior cells ({interior}) for goal, "
                f"{self.n_clutter} clutter, and {num_agents} agents."
            )

        obs_shape = (self.view_px, self.view_px, 3)
        self.action_spaces: List[Space] = [Discrete(NUM_ACTIONS)] * num_agents
        self.observation_spaces: List[Space] = [
            Box(0.0, 1.0, obs_shape, dtype=jnp.float32) for _ in range(num_agents)
        ]

    @property
    def name(self) -> str:
        return "FindGoal"

    def observation_space(self, agent_idx: int) -> Space:
        return self.observation_spaces[agent_idx]

    def action_space(self, agent_idx: int) -> Space:
        return self.action_spaces[agent_idx]

    def _sample_adv_mask(self, key: PRNGKeyArray) -> BoolArray:
        if self.num_adversaries == 0:
            return jnp.zeros((self.num_agents,), dtype=bool)
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, self.num_agents)
        adv_indices = perm[: self.num_adversaries]
        return jnp.isin(jnp.arange(self.num_agents), adv_indices)

    def _generate_episode(self, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        width = height = self.grid_size
        grid = _make_border_grid(width, height)
        coords = _interior_coords(width, height)

        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, coords.shape[0])
        shuffled = coords[perm]

        if self.randomize_goal:
            goal_pos = shuffled[0]
        else:
            goal_pos = jnp.array([width - 2, height - 2], dtype=jnp.int32)

        clutter = shuffled[1 : 1 + self.n_clutter]
        agent_coords = shuffled[
            1 + self.n_clutter : 1 + self.n_clutter + self.num_agents
        ]

        grid = grid.at[goal_pos[0], goal_pos[1]].set(CELL_GOAL)
        grid = grid.at[clutter[:, 0], clutter[:, 1]].set(CELL_WALL)

        return grid, goal_pos, agent_coords

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[FloatArray, State]:
        key, k_ep, k_adv = jax.random.split(key, 3)
        grid, goal_pos, agent_coords = self._generate_episode(k_ep)
        adv_mask = self._sample_adv_mask(k_adv)

        state = State(
            grid=grid,
            goal_pos=goal_pos,
            agent_pos=agent_coords,
            agent_dir=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            agent_done=jnp.zeros((self.num_agents,), dtype=bool),
            agent_active=jnp.ones((self.num_agents,), dtype=bool),
            adv_mask=adv_mask,
            step=0,
            done=jnp.array(False),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> FloatArray:
        """Per-agent POV images, shape (num_agents, view_px, view_px, 3)."""
        return jax.vmap(
            lambda i: _render_agent_pov(
                state.grid,
                state.agent_pos,
                state.agent_active,
                i,
                self.view_size,
                self.view_tile_size,
            )
        )(jnp.arange(self.num_agents))

    def _get_selfpos(self, state: State) -> FloatArray:
        return state.agent_pos.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: PRNGKeyArray,
        state: State,
        actions: IntArray,
    ) -> Tuple[FloatArray, State, FloatArray, BoolArray, dict]:
        key, k_perm = jax.random.split(key)
        order = jax.random.permutation(k_perm, self.num_agents)

        agent_pos = state.agent_pos
        agent_dir = state.agent_dir
        agent_done = state.agent_done
        agent_active = state.agent_active
        step_rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        def process_agent(i: int, carry):
            pos, direction, done, active, rewards = carry
            action = actions[i]
            is_active = active[i]
            already_done = done[i]

            def do_move(c):
                p, d, dn, ac, rw = c
                new_dir = jnp.where(action < ACTION_STAY, action, d[i])
                fwd = p[i] + DIR_VECS[action]
                can_move = _can_move_to(state.grid, p, ac, i, fwd, self.num_agents)
                should_move = (
                    is_active & ~already_done & (action < ACTION_STAY) & can_move
                )
                new_pos_i = jnp.where(should_move, fwd, p[i])
                p = p.at[i].set(new_pos_i)
                d = d.at[i].set(new_dir)

                on_goal = should_move & (
                    state.grid[new_pos_i[0], new_pos_i[1]] == CELL_GOAL
                )
                dn = dn.at[i].set(dn[i] | on_goal)

                rwd = jnp.where(on_goal, self.goal_reward, 0.0).astype(jnp.float32)
                if self.team_reward_type == "share":
                    rw = rw + jnp.where(~state.adv_mask, rwd, 0.0)
                    rw = rw + jnp.where(state.adv_mask, -rwd, 0.0)
                else:
                    rw = rw.at[i].set(rw[i] + rwd)

                stay_active = jnp.where(
                    self.active_after_done,
                    ac[i],
                    ac[i] & ~on_goal,
                )
                ac = ac.at[i].set(jnp.where(is_active, stay_active, ac[i]))
                return p, d, dn, ac, rw

            return jax.lax.cond(
                is_active, do_move, lambda c: c, (pos, direction, done, active, rewards)
            )

        agent_pos, agent_dir, agent_done, agent_active, step_rewards = (
            jax.lax.fori_loop(
                0,
                self.num_agents,
                lambda idx, carry: process_agent(order[idx], carry),
                (agent_pos, agent_dir, agent_done, agent_active, step_rewards),
            )
        )

        step_rewards = _finalize_rewards(
            step_rewards,
            agent_done,
            state.adv_mask,
            self.team_reward_type,
            self.team_reward_multiplier,
        )

        next_step = state.step + 1
        non_adv_mask = ~state.adv_mask
        nonadv_done = jnp.all(jnp.where(non_adv_mask, agent_done, True))
        timeout = next_step >= self.max_steps
        done = jnp.logical_or(state.done, jnp.logical_or(timeout, nonadv_done))

        next_state = State(
            grid=state.grid,
            goal_pos=state.goal_pos,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_done=agent_done,
            agent_active=agent_active,
            adv_mask=state.adv_mask,
            step=next_step,
            done=done,
        )

        info = {}
        if self.observe_self_position:
            info["selfpos"] = self._get_selfpos(next_state)

        return self.get_obs(next_state), next_state, step_rewards, done, info

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
