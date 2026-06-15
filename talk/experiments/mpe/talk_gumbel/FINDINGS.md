# Talk-Gumbel vs TarMAC: Hypothesis Findings

MPE `simple_spread`, `comm_range=-1`, 3 seeds, 1e7 steps, `custom_name=_new_mpe`.
Metric = mean episode return over final 10% of training (higher = better).
TarMAC is the continuous-channel baseline we are trying to match/beat.

## TL;DR

- The original Talk-Gumbel gap to TarMAC was **mostly a training-stability problem, not a
  capacity problem**: with `len_aux_coef=0` the discrete channel **collapses to permanent
  silence in ~2/3 of seeds** (an absorbing state), dragging the seed-mean down.
- Forcing at least one content token (**`min_msg_len=1`**, i.e. mask EOS at `t=0`)
  **eliminates the collapse** in every seed and recovers most of the gap.
- With `min_msg_len=1` the discrete channel **matches TarMAC exactly at low sight (0.25)**,
  where communication matters most, and trails by a modest **~3–4 reward** at higher sight.
- The residual gap is **robust** to Gumbel temperature and attention width, so it is
  intrinsic to the discrete/autoregressive channel (H3/H5), not a tuning artifact.

## Baseline (before any fix), `len_aux_coef=0`

| sight | TarMAC | Talk (coef=0) | seed spread (Talk) |
|------:|-------:|--------------:|--------------------|
| 0.25  | -51.34 | -57.92        | bimodal: {-50.97, -61.05, -61.39} |
| 0.50  | -39.22 | -45.40        | bimodal: one ~-42.5 communicating, rest collapsed |

The seed-mean hides a **bimodal outcome**.

## H2 — Communication collapse to silence (CONFIRMED, dominant cause)

Triaged directly from the existing `coef=0` run metrics:

| seed (sight 0.25) | returns | silence_rate | msg_len | comm_ctx_norm | token_entropy |
|------|--------:|-------------:|--------:|--------------:|--------------:|
| 0 | -61.05 | **1.00** | 0.00 | 0.0001 | 0.002 |
| 1 | **-50.97** | 0.04 | 9.58 | 3.35 | 1.85 |
| 2 | -61.39 | **1.00** | 0.00 | 0.0001 | 0.005 |

Mechanism: early in training, untrained messages are noise that *hurts* the shared team
reward, so the policy gradient drives every agent to emit `EOS` at `t=0`. Once everyone is
silent, `comm_ctx≈0` forever → no learning signal to ever restart → **absorbing state**.
The one seed that stays alive (seed 1) **matches TarMAC** (-50.97 vs -51.34), proving the
channel itself is adequate at this setting; the problem is reaching/keeping the
communicating regime.

## H1 — Gumbel sampling noise / temperature (REJECTED as a fix)

`gumbel_tau=0.5` (sharper, less stochastic messages), sight 0.5:

| config | returns | silence_rate | msg_len |
|--------|--------:|-------------:|--------:|
| `tau=0.5` (no min length) | **-47.90** | **1.00** | 0.00 |

Lower temperature did **not** prevent collapse — it collapsed *all* seeds. Sampling noise is
not the driver; the silence absorbing-state is. Temperature is therefore not a fix on its own.

## The fix — `min_msg_len=1` (mask EOS at t=0)

sight 0.5, `len_aux_coef=0`:

| config | returns | std | silence_rate | msg_len |
|--------|--------:|----:|-------------:|--------:|
| baseline (coef=0) | -45.40 | 2.14 | bimodal | bimodal |
| **`min_msg_len=1`** | **-42.35** | **0.04** | 0.04 | 5.4 |
| TarMAC | -39.22 | 0.14 | – | – |

Forcing ≥1 content token removes the silence absorbing state: **all 3 seeds communicate**,
variance collapses (2.14 → 0.04), and the seed-mean improves by ~3. It does not fully reach
TarMAC (residual ~3.1 at this sight).

## H3 / H5 — Residual gap is intrinsic to the discrete AR channel (supported)

Holding `min_msg_len=1`, sight 0.5:

| config | returns | std | msg_len |
|--------|--------:|----:|--------:|
| `min_msg_len=1` (base, tau=1, attn=32) | -42.35 | 0.04 | 5.4 |
| `+ attn_dim=64` (more bandwidth) | -42.01 | 0.70 | 4.3 |
| `+ gumbel_tau=0.5` | -42.89 | 0.37 | 1.1 |

All three land at ~-42 → the residual gap is **insensitive to attention width and
temperature**. It reflects the lower bandwidth / biased straight-through-Gumbel gradient of a
discrete autoregressive channel relative to TarMAC's continuous vector, rather than a
hyperparameter we left on the table.

## Full sweep vs TarMAC (`min_msg_len=1`, `comm_range=-1`)

| sight | TarMAC | Talk `min_msg_len=1` | gap | verdict |
|------:|-------:|---------------------:|----:|---------|
| 0.25  | -51.34 | **-51.46** | **-0.12** | **matches TarMAC** |
| 0.50  | -39.22 | -42.35 | -3.13 | trails modestly |
| 1.50  | -31.59 | -35.91 | -4.32 | trails |
| -1.0  | -24.81 | -28.59 | -3.78 | trails |

Pattern: the discrete channel **matches TarMAC exactly when sight is smallest** (communication
is the binding constraint and a few discrete tokens carry the missing information). As sight
grows, agents already observe more and TarMAC's continuous channel conveys the finer residual
information that discrete tokens cannot, so the gap opens to ~3–4 reward.

## Overall conclusion

1. The headline "Talk-Gumbel < TarMAC" finding was **largely a collapse artifact**. The single
   most impactful change is `min_msg_len=1` (mask EOS at the first decode step), which makes
   training reliable across seeds.
2. With that fix, Talk-Gumbel **matches TarMAC at low sight** and trails by a modest, robust
   ~3–4 reward elsewhere — we **match under the right conditions** but do not yet beat it.
3. The remaining gap is a genuine discrete-vs-continuous channel-capacity effect, not a tuning
   issue. A deeper cause: in Talk the message *content* read by a receiver is a fixed-codebook
   embedding selected by the sender, so receiver-side error only reaches the sender's hidden
   state through a biased, high-variance straight-through path over a static dictionary — there
   is no direct continuous content highway as in TarMAC. Closing the gap likely requires a
   channel-level change (e.g. emitting continuous, hidden-conditioned payloads addressed by the
   discrete tokens; larger vocab/length budget; or a lower-variance discrete estimator).

### Recommended default
`min_msg_len=1`, `gumbel_tau=1.0`, `attn_dim=32`, `vocab_content=10` — simplest config with the
best stability/return tradeoff observed.

### Caveats
- `min_msg_len=1` removes the "optional silence" property by construction; if silence must stay
  available, a softer anti-collapse mechanism (message-entropy bonus, or annealed `min_msg_len`)
  is untested and worth trying.
- The fresh same-code `baseline_c0` re-run OOM'd under 3-way GPU sharing; the baseline numbers
  above are from the pre-existing `coef=0` runs (identical settings; `min_msg_len` defaults to 0
  and does not change behavior).
- The `sight=-1.0` sweep point reached ~9.99M/10M steps before the process was interrupted;
  its value is effectively converged.
