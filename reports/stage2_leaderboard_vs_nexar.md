# Stage 2: leaderboard vs per-source validation (2026-09-25)

Hypothesis (user): the Stage 2 leaderboard evaluation set resembles NEXAR (long clips, many frames),
because NEXAR validation scores are close to the leaderboard scores.

## Data

Leaderboard (Stage 2): v1 E3 ASFormer = **0.4410**, v3 P2 4-seed ensemble refit on 349 clips = **0.4618**.

Validation = the 70-clip val split (official offline score: 0.35·entry@0.3s + 0.35·collision@0.3s + 0.15·side F1 + 0.15·evasion F1),
95% bootstrap CIs (2000 resamples). v1 head = `E3-ASFormer-349/fixed/seed0` (trained on 279 clips; exactly the submitted head).
v3 was refit on all 349 clips including val, so its own val score is contaminated; the proxy is the identical recipe trained on
279 clips (`ENS_P2_seed0123`, = submit v2).

| Subset | n | v1 E3 (LB 0.441) | v3 proxy P2 ens (LB 0.462) |
|---|---:|---|---|
| ALL | 70 | 0.689 [0.625, 0.750] | 0.752 [0.691, 0.811] |
| AIHub | 18 | 0.663 [0.516, 0.791] | 0.774 [0.646, 0.880] |
| CCD | 17 | 0.696 [0.583, 0.799] | 0.766 [0.679, 0.838] |
| MM-AU | 20 | 0.815 [0.729, 0.892] | 0.898 [0.827, 0.954] |
| **NEXAR** | 15 | **0.522 [0.382, 0.650]** | **0.500 [0.360, 0.639]** |
| non-NEXAR | 55 | 0.734 [0.665, 0.795] | 0.820 [0.764, 0.869] |

NEXAR components (v1 / v3 proxy): entry 0.333 / 0.333, collision 0.667 / 0.467, side F1 0.661 / 0.866, evasion F1 0.489 / 0.603.

Paired bootstrap of the model difference (v3 proxy − v1); the leaderboard difference is **+0.021**:

| Subset | Δ val | 95% CI | P(Δ ≥ +0.021) |
|---|---:|---|---:|
| NEXAR (15) | −0.022 | [−0.143, +0.104] | 0.25 |
| non-NEXAR (55) | +0.086 | [+0.017, +0.151] | 0.97 |
| ALL (70) | +0.063 | [+0.002, +0.122] | 0.91 |

## Verdict: supported on level, not confirmed

* **Supports it (level):** NEXAR is the only source whose score is near the leaderboard. Both LB scores sit inside the NEXAR
  CIs and outside every other source's CI and the non-NEXAR CI (lowest non-NEXAR lower bound 0.516, AIHub/v1).
  A test set mostly made of AIHub/CCD/MM-AU-like clips would be expected to score ≥0.66.
* **Does not confirm it (direction):** on the leaderboard v3 beat v1 by +0.021, while on NEXAR the proxy is 0.022 *worse*
  (collision 10/15 → 7/15). The non-NEXAR slices predict the correct sign. This is weak counter-evidence only:
  15 NEXAR clips give a ±0.12 CI on the difference (P = 0.25 of a ≥+0.021 gain even if the test were NEXAR-like),
  and the submitted v3 is the 349-clip refit, not the proxy.
* **Level is not specific to "long":** any clip set that is hard for these models (different country/camera, different
  labeling convention for entry/collision, unusual frame rate vs the 0.3 s tolerance) would also land near 0.45. The two
  LB numbers cannot separate "long NEXAR-like clips" from "short but out-of-domain clips". Only NEXAR has long clips in
  validation (540–1248 frames); source and length are fully confounded (AIHub 150, CCD 50, MM-AU 50–415).

## How to test it more directly (costs leaderboard submissions)

1. **Length probe:** submit v3 with Stage 2 predictions replaced by a fixed fraction of the clip
   (e.g. collision = entry = round(0.6·N)). On validation, compute the same rule's score per source; the source whose
   rule-score matches the LB result is the best match. Rule-based position priors differ strongly between 50-frame CCD
   (collision ≈ frame 30–40 of 50) and ~1200-frame NEXAR clips.
2. **Long-clip specialist A/B:** submit a model that differs from v3 only on clips with N > 500 frames (e.g. the exact-ENTRY
   variant from `stage2/long_video_experiments`, which changed NEXAR ENTRY 5/15 → 7/15). If the LB moves, the test set has
   long clips; if it is bit-identical, it has none.
3. More NEXAR validation clips (the 80 labeled NEXAR clips exist; grouped CV over them gives n = 80 instead of 15).

## Update 2026-09-26: new leaderboard results

| Submission (Stage 2) | offline fixed all / NEXAR (15) | CV all / NEXAR (80) | LB S2 |
|---|---|---|---|
| v1 E3 ASFormer | 0.689 / 0.522 | — | 0.4410 |
| v3 P2 4-seed refit (proxy: 279-clip ensemble) | 0.757 / 0.500 | — | 0.4618 |
| NEXAR specialist (3 heads + NEXAR KDE prior, every clip) | 0.670 / 0.744 | — | 0.437 |
| length-gated hybrid (>500 frames → 9-head specialist, else P2) | 0.802 / 0.710 | — | 0.4154 |
| **v5 LC-v2 15-head ensemble + motion fusion** | 0.767 / 0.619 | 0.769 / 0.670 | **0.5314** |

* The hybrid differs from the P2 ensemble only on clips with >500 native frames, and it lost 0.046 vs v3. So the hidden test
  set does contain long clips, and on them the NEXAR-fitted temporal prior hurts. Selection on the 15 NEXAR val clips overfit.
* v5 (selected with 5-fold CV, NEXAR n = 80, no position prior) gained +0.070. CV over all clips is the protocol that has
  predicted the leaderboard direction; the 15-clip NEXAR slice has not.
