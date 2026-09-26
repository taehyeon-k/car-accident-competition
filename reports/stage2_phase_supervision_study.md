# Stage 2 phase-supervision study (2026-09-26)

Spec: `stage2/stage2_phase_loss_experiments.md`. Code and per-run results: `stage2/phase_study/` (`model.py`, `losses.py`,
`train.py`, `analyze.py`, `grad_summary.py`, `ensemble_test.py`, `batch*.sh`). Summaries: `stage2/phase_study/results/summary_all.json`,
`ensemble_test_*.json`, `grad_cos_summary.json`. Earlier experiments are untouched.

## Bottom line

* Semantic phase supervision gives a **small, consistent, but sub-threshold** gain as an *auxiliary training signal*.
  With 8 CV seeds: +0.005 to +0.007 pooled-OOF score over the matched no-phase control, 4–5/5 folds.
  The +0.01 success threshold is **not** met robustly. It was met at 3 seeds, but those estimates regressed at 8 seeds.
* The gain is **semantic** (shuffled phase targets: −0.003) and comes mostly from **ENTRY** (+5.5 to +7.3 hits out of 349 clips,
  vs +0.6 to +1.5 COLLISION hits).
* Phase should be **auxiliary only**. Every inference-time use of phase is worse than direct decoding. So is feeding phase
  into the event head, supervising it at multiple scales, a weight of 2, the monotonic term and soft boundaries.
* **The old PH recipe's reported gain was seed luck.** 8 new seeds, and the old code rerun unchanged, give 0.690
  (**−0.010** vs control), not 0.709. The v5 ensemble's PH members are a lucky draw: 0.7688, where the PH recipe averages
  0.7595 ± 0.003 over 4 draws.
* **Best practical formulation:** ordinal phase, λ = 1, + transition 0.75, direct weight 1, direct decoding, direct-decoder
  checkpoint selection. It is the best v5 4th-family recipe: CV 0.7636 ± 0.003 vs NT control 0.7568 ± 0.002 and
  old PH recipe 0.7595 ± 0.003. All 4 of its draws beat every control draw.
  It does **not** beat the specific PH checkpoints already in v5 on CV.

## Protocol

* Unchanged: geometry-DINOv3-S dense cache, adaptive FPS-blind sampling, LCPyramid backbone, side/evasion heads, attribute
  loss 0.5/0.5, AdamW 1e-3, OneCycle, batch 4, 30 epochs, patience 7, batch-1 evaluation, v5 motion fusion (ensemble test only).
  Direct loss = NT (Gaussian σ = 1 sampled position). All targets are in sampled-position units; FPS is used only by the offline metric.
* **Checkpoint selection uses direct decoding for every arm.** The old PH runs selected with the structured decoder; that is
  a second confound, tested separately (`PH_repro_structsel`).
* **Noise.** GPU nondeterminism alone moves one run by up to 0.03: same seed, same code, identical for 3 epochs, then diverges.
  So Stage A (fixed split, 1 seed) was replaced by 5-fold CV for every arm: 3 CV seeds (15 runs) plus 2 fixed seeds; the
  control, the categorical reference and both finalists got 8 CV seeds (40 runs).
  **560 training runs in total.**
* Reported: pooled OOF over all 349 clips per CV seed, mean ± std over seeds. Also per-fold wins vs control, and a paired
  clip bootstrap (2000 resamples, seed-averaged) with 95% CI. "Seed-ens" = probability average over the CV seeds.
  Catastrophic = error > 10% of the clip. tr err = mean |transition argmax − GT| in sampled positions.
* The harness reproduces the old code: A0 = old NT recipe (losses match to 5 decimals until nondeterminism diverges), and
  `ensemble_test.py` reproduces the recorded v5 CV numbers exactly (0.7688 / NEXAR 0.6702).

## Results (5-fold CV; control A0 has 8 seeds; **bold rows have 8 seeds**, others 3)

| arm | loss | CV all | NEXAR | >1000 | E acc | C acc | >1000 E cat / C cat | phase acc | tr err E/C | folds won | Δ vs A0 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0** | direct 1 | 0.699±.010 | 0.551 | 0.543 | .571 | .745 | .208 / .159 | — | — | — | — |
| A1 | direct .25 | 0.705±.003 | 0.533 | 0.534 | .573 | .749 | | — | — | 3/5 | +.006 [−.006,+.018] |
| **A2** | direct 1 + CAT 1 | 0.706±.007 | 0.550 | 0.547 | .589 | .747 | .190 / .159 | .815 | 12.4/8.7 | **5/5** | +.007 [−.002,+.015] |
| A3 | direct .25 + CAT 1 | 0.710±.006 | 0.554 | 0.545 | .599 | .735 | | .836 | 11.4/7.7 | 4/5 | +.010 [−.003,+.023] |
| A4 | A2 + tr .75 + mono .05 | 0.702±.004 | 0.545 | 0.544 | .585 | .736 | | .826 | 9.2/7.0 | 2/5 | +.003 |
| A5 | A3 + tr .75 + mono .05 (old PH loss) | 0.697±.000 | 0.550 | 0.546 | .570 | .734 | | .822 | 8.4/7.6 | 2/5 | −.002 |
| **PH_repro** | A5 + structured-decoder selection (= old PH) | 0.690±.007 | 0.526 | 0.515 | .569 | .714 | .190 / .162 | .833 | 8.7/7.1 | 1/5 | **−.010** [−.020,+.001] |
| E1 shuffled | A2 with misaligned phase targets | 0.696±.005 | 0.556 | 0.552 | .564 | .745 | | .534 | 31/45 | 2/5 | −.003 |
| E1 detach | A2, phase head on h.detach() | 0.704±.004 | 0.538 | 0.532 | .575 | .755 | | .676 | 14.1/9.4 | 3/5 | +.005 |
| **R_ord1** | direct 1 + ORD 1 | 0.705±.010 | 0.561 | 0.556 | .587 | .747 | **.179** / .161 | .820 | 13.0/9.8 | 4/5 | +.005 [−.004,+.015] |
| R_cat soft 1 / 2 | CAT, soft boundaries | 0.700 / 0.702 | .554/.559 | .549/.550 | | | | .82/.81 | | 3/5, 3/5 | +.001 / +.002 |
| R_ord soft 1 / 2 | ORD, soft boundaries | 0.705 / 0.701 | .570/.551 | .565/.547 | | | | .81/.82 | | 4/5, 2/5 | +.006 / +.001 |
| L_ord λ 0.1 / 0.25 / 0.5 | ORD weight sweep | .706 / .707 / .706 | .545/.569/.556 | .537/.558/.553 | | | | .73/.78/.79 | | 3, 5, 4 /5 | +.007 / +.008 / +.007 |
| L_ord λ 2 | | 0.701±.007 | 0.555 | 0.545 | .584 | .741 | | .828 | 13.2/8.6 | 2/5 | +.001 |
| T_ord tr .25 | ORD 1 + transition .25 | 0.710±.005 | 0.574 | 0.567 | .599 | .750 | | .835 | 9.7/8.0 | 4/5 | +.010 |
| **T_ord tr .75** | ORD 1 + transition .75 | 0.706±.008 | 0.547 | 0.537 | .592 | .749 | .205 / .164 | .827 | **9.3/7.7** | 4/5 | +.007 [−.004,+.017] |
| M_ord mono .05 / .2 | ORD 1 + monotonic | .701 / .695 | .559/.553 | .554/.545 | | | | .82/.82 | | 2/5, 1/5 | +.001 / −.005 |
| S2 multiscale | ORD 1, heads on every pyramid level | 0.701±.006 | 0.566 | 0.564 | .568 | .755 | | .795 | 13.8/10.4 | 2/5 | +.001 |
| S3 cond detach / e2e | event head also sees local phase | .698 / .704 | .547/.568 | .541/.554 | | | | .81/.81 | | 2/5, 4/5 | −.002 / +.005 |

Winner's curse: 3-seed arms that looked best regressed at 8 seeds (R_ord1 +.010 → +.005; T_ord tr .75 +.010 → +.007).
Treat 3-seed differences below about 0.01 as unresolved.

Seed-ensemble CV (probability average of the CV seeds): A0 .729, A2 .734, R_ord1 .741, **T_ord tr .75 .745**, PH_repro .719.

### Decoders (same checkpoints, 8 seeds; D0 = direct)

| model | D0 direct | D1 transition | D2 local re-rank K4 / K8 (w from other folds) | D3 global structured |
|---|---|---|---|---|
| A2 CAT | **0.706** | 0.652 | 0.703 / 0.701 | 0.632 |
| R_ord1 | **0.705** | 0.636 | 0.698 / 0.697 | 0.626 |
| T_ord tr .75 | **0.706** | 0.679 | 0.702 / 0.702 | 0.654 |
| PH_repro | **0.690** | 0.681 | 0.690 / 0.690 | 0.666 |

D2 needs no whole-clip accumulation and is the least harmful phase decoder. Its weight, chosen on other folds, almost always
goes to the smallest value (0.25), i.e. "use as little phase as possible". D2 never beats D0. Even with a transition loss,
D1/D3 lose 0.03–0.08 and add catastrophic ENTRY errors (Ecat 0.19 → 0.20–0.28).

### Gradient interaction (shared pyramid parameters, 8 fixed train batches, every epoch)

| λ (ORD) | cos(direct, phase) init → ep1-3 → ep4-10 → ep11+ | abs(g_phase)/abs(g_direct), ep4-10 |
|---|---|---|
| 0.1 | −.04 → +.14 → +.27 → +.22 | 1.37 |
| 0.25 | −.04 → +.15 → +.26 → +.26 | 1.12 |
| 0.5 | −.04 → +.18 → +.33 → +.35 | 0.77 |
| 1 | −.04 → +.22 → +.39 → +.46 | 0.63 |
| 2 | −.04 → +.27 → +.46 → +.56 | 0.44 |

The objectives are **cooperative** at every λ. The cosine starts at 0 and grows positive as training shapes a shared
representation, and grows with λ. Transition gradients are also aligned with direct gradients (+.3 to +.6). So λ = 2's negative
transfer is not a gradient conflict: the weighted phase gradient (2 × 0.44 ≈ 0.9 of the direct gradient) dominates the shared
update, and the model optimises phase labelling at the expense of precise event spotting.

### v5 ensemble test (C0 + X_ema + M_motion unchanged + one 4th family; 8 members per fold; v5 motion fusion + snap)

4th family drawn as 4 disjoint 2-seed pairs (seeds {0,1}, {2,3}, {4,5}, {6,7}); CV β chosen on other folds; fixed split uses the 4th family's seeds 0–1 and β = 1.

| 4th family | CV all (mean ± std, 4 draws) | NEXAR | >1000 | fixed all / NEXAR |
|---|---|---|---|---|
| NT control (A0) | 0.7568 ± 0.0019 | 0.638 | 0.631 | 0.746 / 0.557 |
| old PH recipe (PH_repro) | 0.7595 ± 0.0029 | 0.640 | 0.631 | 0.758 / 0.627 |
| CAT (A2) | 0.7600 ± 0.0060 | 0.637 | 0.630 | 0.744 / 0.534 |
| ORD (R_ord1) | 0.7594 ± 0.0024 | 0.645 | 0.636 | **0.779 / 0.604** |
| **ORD + transition .75** | **0.7636 ± 0.0029** | **0.650** | **0.642** | 0.753 / 0.589 |
| *v5's actual PH members (old seeds 0,1)* | *0.7688 (1 draw)* | *0.670* | *0.660* | *0.762 / 0.581* |
| *old NT seeds 0,1 (the v5 same-size control)* | *0.7510 (1 draw)* | *0.641* | *0.632* | *0.759 / 0.581* |

The recorded "+0.018 from PH over NT in v5" combined a lucky PH draw with an unlucky NT draw. The PH draw (0.7688) is above
all 4 PH-recipe draws (max 0.7623) and above 20 of the 21 other draws in this table (only one CAT draw, 0.7702, is higher). The NT
draw (0.7510) is below all 4 A0 draws. At the recipe level, ORD + transition adds +0.007 over the no-phase control and +0.004 over the old PH
recipe.

## Answers to the required questions

1. **Is the benefit caused by semantic phase supervision?** Yes, as far as it exists. True targets +0.007 (8 seeds); shuffled
   targets −0.003 (a +0.010 gap), although the shuffled head still learns plausible phases (acc .53 vs .82). Detaching the head
   removes the ENTRY gain (E acc .575 vs .589, phase acc .68 vs .82), so the effect comes from shaping the shared representation.
   The detached arm's overall +0.005 is within noise.
2. **Was the previous gain caused by the direct-loss weight?** Partly, but mainly it was not a real gain. The direct weight
   0.25 alone gives +0.006 (A1), phase alone +0.007 (A2), both +0.010 (A3), so the effects are roughly independent and neither is
   significant. The old composite recipe is actually worse than the control (−0.010 with 8 seeds; old code on new seeds agrees: 0.687/0.689).
   Of that loss, ≈0.008 comes from **selecting checkpoints with the structured decoder** (A5 −0.002 → PH_repro −0.010, almost all
   COLLISION: −10.8 hits), and the rest from transition + mono at direct weight 0.25. The reported 0.709 was two lucky seeds.
3. **Categorical vs ordinal vs soft?** CAT ≈ ORD on the aggregate (8 seeds: .706 vs .705). ORD is better where it matters:
   NEXAR .561 vs .550, >1000 .556 vs .547, >1000 ENTRY catastrophic .179 vs .190, seed-ensemble .741 vs .734, and it keeps
   q_E ≥ q_C by construction. **Soft boundaries hurt both**, monotonically with width: the exact sampled transition is the useful signal.
4. **Best phase weight?** A plateau from λ = 0.1 to 1 (+0.007 to +0.008 at 3 seeds). λ = 2 gives negative transfer (+0.001, more
   catastrophic long-clip collisions). Use 0.25–1; λ = 1 has the best phase accuracy and ensemble behaviour.
5. **Transition and monotonic terms independently?** Transition: no single-model gain (Δ ≈ 0 vs ORD alone), but better transition
   localisation (tr err 9.3/7.7 vs 13.0/9.8) and **better ensemble behaviour** (seed-ens .745, v5 +0.004 over ORD alone). Monotonic:
   **harmful** (−0.007 at .05, −0.013 at .2 vs ORD alone; the second CI excludes 0). Don't combine them.
6. **Multi-scale phase supervision?** No: −0.009 vs the final-level head (S1), with lower phase accuracy.
7. **Auxiliary-only, event-conditioned, or decoding?** **Auxiliary only.** Event conditioning: detached −0.012 (CI excludes 0) and
   end-to-end −0.005 vs S1. Decoding: D1 −0.03 to −0.07, D3 −0.05 to −0.08, local re-rank D2 −0.003 to −0.008.
8. **ENTRY, COLLISION, or both?** Mainly **ENTRY**: +5.5 to +7.3 ENTRY hits vs +0.6 to +1.5 COLLISION hits (8-seed arms vs A0).
9. **Fewer catastrophic errors on long videos?** Only for ENTRY and only modestly: >1000 ENTRY catastrophic .208 → .179 (ORD) /
   .190 (CAT). COLLISION catastrophic is unchanged (.159–.164). Phase does not fix wrong-event COLLISION selection.
10. **Does the best formulation improve the v5 ensemble?** At the recipe level, yes, a little: ORD + transition .75 as the 4th
    family gives 0.7636 ± 0.003 vs 0.7595 (old PH recipe) and 0.7568 (NT), with every draw above every NT draw. It does **not**
    beat the particular PH checkpoints in v5 on CV (0.7688). Their advantage is draw-selection luck on the same OOF clips,
    and it is not expected to transfer, but it cannot be ruled out offline.

### Success criteria (8-seed finalists vs matched control A0)

| criterion | A2 CAT | R_ord1 | T_ord tr .75 |
|---|---|---|---|
| ≥ +0.01 OOF | ✗ (+.007) | ✗ (+.005) | ✗ (+.007) |
| ≥ 4/5 folds | ✓ 5/5 | ✓ 4/5 | ✓ 4/5 |
| >1000 not −0.01 | ✓ +.004 | ✓ +.013 | ✓ −.006 |
| improves v5 when replacing PH members | recipe ≈ | recipe ≈ | recipe ✓ (+.004), actual members ✗ |

No formulation passes all criteria. The best is ORD + transition .75: it fails only the +0.01 single-model bar and the comparison with v5's specific lucky PH draw.

## Implications for the next Stage 2 submission (not built, per spec)

* Keep phase as an auxiliary loss: **ORD, λ = 1, transition 0.75, direct weight 1, no mono, final-level head, direct decoding,
  direct-decoder selection**. Never use phase at inference.
* The expected value of any phase change is ≈ +0.005 single-model / +0.004 in the ensemble, within the seed-draw noise of a
  2-seed family. Ensemble size and refitting on all 349 clips are the larger levers. v5 members were trained on 279 clips only.
* Stop judging recipes by 2-seed comparisons; use ≥ 4 disjoint draws (this study's `ensemble_test.py --seed-sets`).
