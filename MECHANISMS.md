# DePose-Fi: Two Systems Mechanisms

This note explains, for co-authors, the two systems-level mechanisms in the
decomposition-guided Wi-Fi HPE pipeline:

1. **Proactive Resource-Aware Rank Adaptation (PRA)** — a runtime controller that
   treats CP-decomposition rank as a knob and picks it *ahead* of each frame's deadline.
2. **Distributed & Parallel Execution** — how the multi-branch S-AFF predictor and the
   CP → S-AFF pipeline map onto parallel / heterogeneous hardware.

They are complementary: PRA decides *how much work* each frame does; parallel execution
decides *how that work is laid out on hardware*. PRA's chosen rank is exactly the input
that sets the pipeline's stage balance.

The full pipeline per frame:

```
CSI frame ──▶ CP decomposition (rank R) ──▶ factors A,B,C ──▶ S-AFF ──▶ pose
             (iterative ALS, the cost knob)   (fork-join)     (gate+decode)
```

---

## Part A — Proactive Resource-Aware Rank Adaptation (PRA)

### The problem
CP decomposition at rank `R` trades accuracy against latency. Higher `R` → more
components → better pose, but ALS cost grows ~linearly in `R`. On a shared edge device
the available CPU fraction `ρ` fluctuates (other processes, thermal throttling), so the
*effective* latency of a fixed rank is `L(R)/ρ`. A single fixed rank is therefore wrong
in two ways:

- **Fixed low rank** (e.g. R=2): never misses the deadline, but leaves accuracy on the
  table whenever the CPU is idle.
- **Fixed high rank** (e.g. R=16): great when the CPU is free, but blows the deadline
  the moment contention spikes — and a missed frame delivers *no* pose.

Reacting *after* a miss is also poor: by the time you detect an overrun, the frame is
already late.

### The idea
Treat `R` as a per-frame runtime knob. **Forecast** near-term CPU availability and
**proactively** select the highest rank whose predicted effective latency still fits the
deadline `D`. One model serves all ranks (see "anytime model" below), so switching rank
costs nothing but a different truncation of the same factors.

### The controller (proactive policy)
Per frame `t`:

1. **Forecast** available CPU with an EWMA of the observed trace:
   `ρ̂ ← α·ρ_{t-1} + (1-α)·ρ̂`  (default `α=0.3`).
2. **Plan against a shrunk deadline** `D_plan = D·(1 - safety_margin)` (default margin
   0.15) so forecast error doesn't immediately cause a miss.
3. **Feasible rank**: `feasible_max_rank = max{ R : L(R)/ρ̂ ≤ D_plan }`.
4. **Asymmetric switching (hysteresis)**: drop rank *immediately* when the current rank
   becomes infeasible (safety first); raise rank only after `hysteresis_frames` (default
   5) consecutive frames of slack (stability, avoids thrashing).

This is implemented in `run_policy(...)` in `experiments/exp30_pra_contention.py`
(PCK metric) and `experiments/exp33_pra_mpjpe_contention.py` (MPJPE metric).

### Baseline policies we compare against
- `fixed_min` / `fixed_max`: constant lowest / highest rank.
- `reactive`: lower rank after an observed overrun, raise after sustained slack — no
  forecast.
- `proactive`: the controller above.
- `oracle`: knows the *true* `ρ_t` and picks the best feasible rank — upper bound.

### The two models the controller needs
- **Accuracy model `A(R)`** — measured, not assumed. We train ONE *anytime* CP+S-AFF
  model with component-count dropout over candidate ranks (e.g. `{2,4,8,12,16}`), energy-
  sort the CP components per frame, and evaluate the model at each rank prefix. This
  yields per-frame accuracy at every rank from a single model
  (`experiments/exp29_anytime_cp_saff.py` for MM-Fi, `exp31_piw_anytime_pra.py` for PiW).
- **Latency model `L(R)`** — `L(R) = cp_count · cp_us_at_rank4 · (R/4) · (iters/10) + saff_us`.
  `cp_count` = number of CP extractions per frame (single-CP sanitized-phase = 1,
  dual-CP amp+phase = 2). Real per-rank costs can be injected via `--latency-json` from
  the on-device calibration (`exp34_dualcp_latency_calib.py`). **The anchor is currently a
  placeholder and must be calibrated before any headline latency claim.**

### How quality is scored under contention
A missed-deadline frame must be charged its true cost, not a free pass:
- PCK sim (`exp30`): missed frame scored **0** (worst PCK).
- MPJPE sim (`exp33`): missed frame charged the **mean-pose baseline error** (i.e. "no
  valid pose delivered this slot"), because 0 mm would wrongly look *perfect*.

Reported per policy: `drop_rate`, delivered quality, on-time quality, mean rank, number
of switches, mean effective latency.

### When PRA actually wins (the honest condition)
PRA pays off only when **`A(R)` is steep** (rank genuinely buys accuracy) **and** the CPU
**fluctuates** **and** the deadline has **enough slack** to occasionally ride high ranks.

- **MM-Fi**: `A(R)` is nearly flat (PCK20 r2→r8 only +1.5). Proactive beats reactive and
  fixed_max and is far more stable, but **loses narrowly to trivial fixed_r2** — the
  cheapest rank never misses and is almost as accurate. No deadline setting rescues it.
- **Person-in-WiFi-3D**: `A(R)` is steeper, so proactive beats fixed and reactive and sits
  near oracle. **The win grows as the deadline loosens** (more room to opportunistically
  use high ranks when the CPU is free), *not* as it tightens — under a truly tight
  deadline every policy is pinned near the minimum rank.

### Results status (read before citing numbers)
- **MM-Fi PRA (PCK): valid.** `results/pra_D*.csv`, `PAPER/figures/fig_pra_*.png`.
- **PiW PRA (PCK): SUPERSEDED.** Our PCK normalization was tuned for MM-Fi, not PiW, so
  the PiW PCK-based PRA numbers (`results/piw_pra_*.csv`) are **not valid** and are being
  redone in **MPJPE only** (`exp33`, outputs `results/piw_mpjpe_D*.csv` + stratified
  `piw_anytime_ranks_by_people.csv`). That rerun is in progress at the time of writing.
- **Two validations still owed before publication:** (1) real per-rank on-device CP
  latency calibration (`exp34`); (2) a *dedicated-per-rank* baseline, to prove the anytime
  model's accuracy tax does not erase the PRA win.

---

## Part B — Distributed & Parallel Execution

Full derivation and the two algorithm blocks are in the paper draft
(`PAPER/deposefi_systems_draft.tex`, subsection "Distributed and Parallel Execution",
`\label{sec:parallel}`). Summary for co-authors:

### The multi-branch structure is a fork–join
S-AFF is **not** a monolithic MLP. After CP produces factors
`A ∈ R^{L×R}`, `B ∈ R^{S×R}`, `C ∈ R^{P×R}`, three mode-specialized encoders read
**disjoint** slices with no shared weights or activations:

```
        ┌─ g_A(A) = f_A ─┐
CP ─────┼─ g_B(B) = f_B ─┤ ─▶ concat ─▶ fuse expert f_F
        └─ g_C(C) = f_C ─┘        └────▶ gate α = softmax(W_g·[f_A,f_B,f_C]/τ)
                                          │
                          base = Σ_m α_m · proj_m  ─▶ query decoder ─▶ poses
```

- **Fork (parallel):** `f_A, f_B, f_C` are data-independent → run concurrently on 3 lanes.
- **Join (barrier):** fused expert, gate, gated sum, and the query decoder all run
  *after* the three branches complete.

So it is a **map (3 branches) → reduce (fuse/gate/decode)**, not independent end-to-end
pipelines.

### Two honest caveats
1. **Branch parallelism alone barely helps latency.** Speedup of the S-AFF stage is
   bounded by `(w_A+w_B+w_C)/max(w_A,w_B,w_C) ≤ 3×`, realistically ~1.5× because the
   subcarrier branch (two 1-D conv blocks) dominates. And S-AFF is **not** the
   bottleneck — CP-ALS is (`cp_us ≫ saff_us ≈ 250 µs`).
2. **The strong lever is pipeline parallelism across frames.** Double-buffer the two
   stages: while unit U₁ factorizes frame `t+1`, unit U₂ runs S-AFF on frame `t`.
   Throughput goes from `1/(T_CP + T_SAFF)` to `1/max(T_CP, T_SAFF)`; since CP dominates,
   S-AFF is almost fully **hidden** behind the next frame's factorization. Branch-level
   fork–join only becomes worthwhile once CP itself is accelerated/offloaded (or replaced
   by a learned feed-forward factor extractor), which puts S-AFF back on the critical path.

### Other parallelism handles
- **Heterogeneous mapping:** conv-heavy subcarrier branch → NPU/DSP; dense link & packet
  branches → CPU; they overlap on distinct units.
- **Data-parallel CP:** the matricized tensor-times-Khatri-Rao products parallelize across
  the `R` components and across the 3 physical links (a 3×3 antenna array in PiW).
- **Batched queries:** the `Q` person queries in the decoder are already one batched
  matmul; coupling is confined to a single small attention mixer.

### How PRA and parallel execution compose
PRA picks rank `R_t` each frame; `R_t` sets both `T_CP` (stage-1 cost) and the branch
input width, so it directly controls the pipeline's stage balance. The pipelined streaming
loop (Algorithm 2 in the draft) calls `SelectRank(ρ̂_t, D)` as its first step.

### Status of the claim
The **structural independence is exact** (a property of the architecture). The resulting
**latency/energy gains are not yet measured on device** — thread-launch / NPU-dispatch
overhead can exceed the tiny branch compute, and the pipeline win depends on the real
`T_CP / T_SAFF` ratio. We will report per-rank on-device CP and S-AFF latencies, single-
lane vs fork–join S-AFF, and sequential vs pipelined throughput. **No speedup is claimed
that has not been timed.**

---

## File map

| Concern | File |
|---|---|
| MM-Fi anytime model (A(R)) | `experiments/exp29_anytime_cp_saff.py` |
| PRA contention sim, PCK | `experiments/exp30_pra_contention.py` |
| PiW anytime model | `experiments/exp31_piw_anytime_pra.py` |
| PiW dual-CP anytime model | `experiments/exp32_piw_dualcp_anytime.py` |
| PRA contention sim, MPJPE | `experiments/exp33_pra_mpjpe_contention.py` |
| On-device CP latency calibration | `experiments/exp34_dualcp_latency_calib.py` |
| Paper writeup (both mechanisms) | `PAPER/deposefi_systems_draft.tex` |
| MM-Fi PRA results (valid) | `results/pra_D*.csv` |
| PiW PRA results (PCK — superseded) | `results/piw_pra_*.csv` |
