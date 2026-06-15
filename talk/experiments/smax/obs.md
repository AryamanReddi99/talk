# SMAX observations (JaxMARL)

This note describes how **JaxMARL** builds observations for [`SMAX`](../../../JaxMARL/jaxmarl/environments/smax/smax_env.py) and how the **talk** SMAX experiments (`mappo`, `mappo_gru`) consume them. The implementation follows the SMAC-style “unit list” partial observability model used in the original PyMARL/SMAC stack, ported to JAX.

**Primary source:** `JaxMARL/jaxmarl/environments/smax/smax_env.py`  
**Talk stack:** `HeuristicEnemySMAX` → `SMAXWorldStateWrapper` → MAPPO actor/critic

---

## Defaults that matter

| Setting | Default | Used in talk (`config_mappo.yaml`) |
|--------|---------|-------------------------------------|
| Observation mode | `"unit_list"` | (implicit default) |
| Map size | 32 × 32 | (implicit) |
| `see_enemy_actions` | `True` | `true` |
| `walls_cause_death` | `True` | `true` |
| `attack_mode` | `"closest"` (heuristic enemies only) | `closest` — **does not change obs** |
| `observation_type` | `"unit_list"` | (implicit) |
| `obs_with_agent_id` | — | `true` (talk wrapper only; critic) |

Enemy behaviour (`attack_mode`) is configured on [`HeuristicEnemySMAX`](../../../JaxMARL/jaxmarl/environments/smax/heuristic_enemy_smax_env.py); it only affects how scripted enemies pick attack actions, not the observation function.

---

## Unit types and sight range

SMAX supports six unit archetypes (indices 0–5). **Sight range is per unit type** and is the Euclidean radius within which observer \(i\) can see another unit \(j\):

\[
\| \mathbf{p}_j - \mathbf{p}_i \|_2 < \texttt{unit\_type\_sight\_ranges}[\texttt{unit\_types}[i]]
\]

Default values from `SMAX.__init__` (game-distance units, same scale as the 32×32 map):

| Index | Name | Shorthand | Sight range | Attack range | Max health | Weapon cooldown (s) |
|------:|------|-----------|------------:|-------------:|-----------:|--------------------:|
| 0 | marine | m | **9.0** | 5.0 | 45 | 0.61 |
| 1 | marauder | M | **10.0** | 6.0 | 125 | 1.07 |
| 2 | stalker | s | **10.0** | 6.0 | 160 | 1.87 |
| 3 | zealot | Z | **9.0** | 2.0 | 150 | 0.86 |
| 4 | zergling | z | **8.0** | 2.0 | 35 | 0.50 |
| 5 | hydralisk | h | **9.0** | 5.0 | 80 | 0.59 |

There is **no global fog-of-war constant**: each controlled unit observes using **its own** type’s sight range. On the talk default map **`2s3z`** (2 stalkers + 3 zealots per side), allies therefore use sight **10** for stalkers and **9** for zealots.

Other combat stats (velocity, damage, collision radius) affect dynamics and action masks but not the observation layout.

---

## Controlled agents vs full environment

- Full `SMAX` has `num_allies + num_enemies` agents (self-play).
- **`HeuristicEnemySMAX`** exposes only **`ally_0 … ally_{N-1}`** to the learner; enemies are driven by a heuristic policy. Each ally still receives the same **local** observation it would get in full SMAX (other allies + all enemy slots), plus a **`world_state`** vector for centralized training (CTDE).

---

## Observation modes

### 1. `unit_list` (default)

Fixed-length vector per agent: **all other units in a fixed order** (ally slots, then enemy slots), then **own-unit features**. This is what talk uses.

Observation size:

```text
obs_dim = (num_allies - 1 + num_enemies) × len(unit_features) + len(own_features)
```

With default feature sets:

- `len(unit_features) = 7 + 6 = 13` (7 scalars + 6-d unit-type one-hot)
- `len(own_features) = 4 + 6 = 10` (4 scalars + 6-d unit-type one-hot)

**Example — map `2s3z`:** 5 allies, 5 enemies → \(4 + 5 = 9\) other-unit blocks:

```text
obs_dim = 9 × 13 + 10 = 127
```

`observation_spaces[agent]` is `Box(low=-1, high=1, shape=(127,))` (some raw features can lie outside \([-1,1]\) in practice, e.g. absolute position on the map).

### 2. `conic` (optional, not used in talk)

Alternative that bins visible units into **32 angular sectors**, up to **2 units per sector** (`num_sections=32`, `max_units_per_section=2`). Same per-unit feature vector inside each slot. Larger and structured for cone-like visibility; enable with `observation_type="conic"`.

---

## `unit_list` layout (detailed)

For observer index `i`, `get_obs_unit_list` builds:

```text
[ other_unit(i, slot_0) ‖ … ‖ other_unit(i, slot_{K-1}) ‖ own_unit(i) ]
```

where \(K = \texttt{num\_allies} - 1 + \texttt{num\_enemies}\).

### Slot ordering

- **Allies** (`i < num_allies`): slots enumerate **other allies first** (fixed ally indices), then **all enemies** in a stable order.
- **Enemies** use a symmetric rule with indices counted in reverse so that “allies first” holds from that team’s perspective.

Slots refer to **logical indices** `j_idx` into the global unit array; the implementation remaps slot index `j` so teammates and enemies always appear in the same positions across episodes for a given map.

### Visibility mask

For each slot, if **any** of the following fails, the entire 13-d block is **zeros**:

1. Observer \(i\) is alive.
2. Observed unit `j_idx` is alive.
3. Euclidean distance \(<\) observer’s sight range (see table above).

There is **no partial visibility** within a slot: either the full feature vector is filled or the block is empty.

---

## Feature vectors

### Other-unit block (`unit_features`, 13 dims)

Built by `_observe_features(state, i, j_idx)` for observer \(i\) viewing unit \(j\):

| Index | Name in code | Meaning |
|------:|--------------|---------|
| 0 | `health` | `unit_health[j] / unit_type_health[type[j]]` ∈ [0, 1] (0 if dead/masked) |
| 1 | `position_x` | \((p_{j,x} - p_{i,x}) / \texttt{sight\_range}_i\) |
| 2 | `position_y` | \((p_{j,y} - p_{i,y}) / \texttt{sight\_range}_i\) |
| 3 | `last_movement_x` | Previous step movement vector, x component (see below) |
| 4 | `last_movement_y` | Previous step movement vector, y component |
| 5 | `last_targeted` | **Previous discrete action index** `prev_attack_actions[j]` (despite the name, not a unit id) |
| 6 | `weapon_cooldown` | Remaining weapon cooldown for unit \(j\) |
| 7–12 | `unit_type_bits_*` | One-hot of `unit_types[j]` (6 types) |

**Relative position** is normalized by the **observer’s** sight range, not the observed unit’s. Components are bounded in magnitude by roughly 1 when the unit is visible (visibility uses full Euclidean distance).

**Previous movement** (`prev_movement_actions`): 2D velocity vector from the last environment step’s decoded movement (diagonal unit vectors scaled by type velocity inside the physics substeps). Movement actions 0–3 are the four diagonals; action 4 is **stop**.

**Previous attack / “last_targeted”**: scalar storing the last step’s **raw discrete action** for unit \(j\) (movement indices 0–4 or attack indices ≥ 5). When `see_enemy_actions=False`, enemy team movement/attack channels are zeroed for cross-team entries.

### Own-unit block (`own_features`, 10 dims)

Built by `_get_own_features(state, i)`:

| Index | Name | Meaning |
|------:|------|---------|
| 0 | `health` | Normalized health of \(i\) |
| 1 | `position_x` | \(p_{i,x} / \texttt{map\_width}\) (absolute, default 32) |
| 2 | `position_y` | \(p_{i,y} / \texttt{map\_height}\) |
| 3 | `weapon_cooldown` | Own cooldown |
| 4–9 | `unit_type_bit_*` | One-hot of own type |

If \(i\) is dead, the own block is **all zeros**.

**Important:** own position is **absolute** and map-normalized; other-unit positions are **relative** and sight-normalized. This matches the JaxMARL/SMAC design choice.

---

## `see_enemy_actions`

When `see_enemy_actions=True` (talk default), an observer sees **enemy** units’ previous movement (dims 3–4) and previous action index (dim 5) the same as allies. When `False`, those three scalars are zero for any unit on the opposing team.

Ally–ally and enemy–enemy pairs always see each other’s action history.

---

## `world_state` (centralized critic)

`get_world_state` concatenates **global** information (no sight mask):

```text
world_state = flatten(own_features for every unit in the battle)
              ‖ unit_teams (uint)
              ‖ unit_types (uint indices)
```

Per-agent own-feature block in world state uses the same 10-d encoding as above (including dead units as zeros). For `N` total units in the underlying `SMAX` instance:

```text
state_size = (len(own_features) + 2) × N = 12 × N
```

For `2s3z`, \(N = 10\) → **120** floats before the talk wrapper.

### Talk: `SMAXWorldStateWrapper` + `obs_with_agent_id`

With `obs_with_agent_id: true` (default in talk configs), each ally’s critic input is:

```text
world_state_for_ally_k = repeat(global_world_state, num_allies)
                       ‖ one_hot(k, num_allies)
```

So critic dimension is `120 + 5 = 125` on `2s3z`. The actor still uses only the **127-d** local observation.

---

## Map `2s3z` (talk default)

From `MAP_NAME_TO_SCENARIO["2s3z"]`:

- **5 allies, 5 enemies**
- Fixed types per slot: `[2,2,3,3,3]` per team → **2 stalkers** (sight 10, range 6) + **3 zealots** (sight 9, melee range 2)

Per-ally local obs: **127** dims.  
Per-ally `avail_actions`: **10** discrete actions (5 move/stop + 5 enemy attack targets), masked by range, cooldown, and alive status (`get_avail_actions`).

---

## Action space (context for obs channels 3–5)

Discrete actions per ally:

| Index | Meaning |
|------:|---------|
| 0–3 | Move along one of four diagonal directions |
| 4 | Stop |
| 5–9 | Attack enemy slot `0 … num_enemies-1` (in-battle indexing) |

Attack slots are **masked** unless the enemy is alive and within **attack range** (type-specific, ≤ sight range). The observation’s dim 5 stores the **last discrete action**, not the attack mask.

---

## Related files

| File | Role |
|------|------|
| `JaxMARL/jaxmarl/environments/smax/smax_env.py` | Observation math, sight ranges, `unit_list` / `conic` |
| `JaxMARL/jaxmarl/environments/smax/heuristic_enemy_smax_env.py` | Ally-only API + `world_state` passthrough |
| `talk/experiments/smax/mappo/mappo.py` | `SMAXWorldStateWrapper`, batching, MAPPO |
| `talk/experiments/smax/mappo/config_mappo.yaml` | `map_name`, `env_kwargs`, `obs_with_agent_id` |

---

## Quick reference diagram (`unit_list`, one ally)

```text
┌─────────────────────────────────────────────────────────────────┐
│  Ally slot 1 (13) │ Ally slot 2 (13) │ … │ Enemy slots (13 each) │
├─────────────────────────────────────────────────────────────────┤
│  health, rel_pos/sight, last_move(2), last_action, cd, type_oh  │  × K slots
├─────────────────────────────────────────────────────────────────┤
│  own: health, abs_pos/map, cd, type_one_hot (10)                 │
└─────────────────────────────────────────────────────────────────┘
         K = (num_allies - 1) + num_enemies
         Empty slot if out of sight, dead, or observer dead
```
