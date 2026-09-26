# Stage 2 complementary-signal study — INTERIM (stopped by request, 2026-09-26)

Spec: `stage2/stage2_complementary_experiments.md`. Code: `stage2/aux_signal_experiments/`. Stopped before the 8-seed finalist
extensions, the ensemble tests of the finalists and the pairwise combinations; **no submission recommendation yet.**

Protocol: unchanged geometry-DINOv3-S cache, adaptive sampling, LCPyramid, AdamW/OneCycle recipe, 5 LC-v2 folds, batch-1 eval,
direct-decoder checkpoint selection, FPS used only by the offline metric. The no-signal arms reproduce `phase_study` A0_d1 exactly
(losses equal to 4–5 decimals), which is the matched control B0 (8 CV seeds). Motion arms use the M_motion (P2) loss, with control
M0 = M_motion recipe. Screening = 5-fold CV × 3 seeds + 2 fixed seeds per arm (single seeds vary by ±0.03 from GPU
nondeterminism alone). Only seeds with all 5 folds finished are counted.

## Results (pooled 5-fold OOF, mean ± std over CV seeds; Δ = paired clip bootstrap vs matched control, 95% CI)

| signal | arm | seeds | CV all | NEXAR | >1000 | E acc | C acc | >1000 C cat | Δ E / C hits | folds | Δ vs control |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **B0 control (NT)** | 8 | 0.699±.010 | 0.551 | 0.543 | .571 | .745 | .159 | — | — | — |
| risk | R1 w .5 / w 1 | 3 | .697 / .697 | .562/.544 | .547/.540 | .562/.567 | .753/.757 | .130/.130 | −3.2/+2.7, −1.2/+4.3 | 3, 1 | −.003 / −.002 |
| risk | R2 future-consistency | 5 | 0.704±.008 | 0.560 | 0.554 | .581 | .756 | .132 | +3.6 / +4.0 | 3 | +.005 [−.005,+.015] |
| risk | R3 adaptive monotonic | **8** | 0.695±.010 | 0.543 | 0.541 | .566 | .745 | .148 | −1.6 / +0.0 | 1 | −.005 [−.013,+.004] |
| boundary | BND1 h_t, w1 / w2 | 3 | .703 / .701 | .553/.553 | .547/.546 | .585/.581 | .750/.741 | .156/.177 | +4.8/+1.7, +3.4/−1.3 | 3, 2 | +.004 / +.002 |
| boundary | **BND2 [h_t, Δh_t], w1** | 3 | **0.712±.009** | 0.564 | 0.558 | .598 | .760 | .121 | **+9.4 / +5.3** | 3 | **+.013 [+.002,+.024]** |
| boundary | BND2, w2 | 3 | 0.707±.012 | 0.570 | 0.561 | .585 | .764 | .147 | +4.8 / +6.7 | 4 | +.008 [−.003,+.018] |
| boundary | BND3 local conv, w1 / w2 | 3 | .704 / .705 | .554/.574 | .550/.567 | .585/.573 | .758/.758 | .117/.143 | +4.8/+4.7, +0.8/+4.7 | 3, 3 | +.005 / +.005 |
| lane (v2 labels) | L1 4-state | 3 | 0.705±.011 | 0.561 | 0.550 | .568 | .759 | .156 | −0.9 / +5.0 | 3 | +.006 [−.006,+.017] |
| lane (v2 labels) | **L2 ordinal intrusion** | 3 | **0.710±.007** | 0.558 | 0.547 | .587 | .761 | .169 | +5.8 / +5.7 | 4 | **+.010 [−.001,+.023]** |
| — | **M0 = M_motion recipe** (control for motion) | 3 | 0.707±.007 | 0.540 | 0.535 | .585 | .759 | .139 | — | — | — |
| residual motion | **M1 global + residual input** | 3 | **0.730±.002** | **0.615** | **0.599** | .589 | **.813** | **.074** | +1.3 / **+18.7** | **4** | **+.023 [+.007,+.039]** vs M0 |
| residual motion | M2 residual only | 3 | 0.719±.010 | 0.617 | 0.603 | .579 | .808 | .058 | −2.3 / +17.0 | 2 | +.012 [−.008,+.030] vs M0 |
| residual motion | M3 late decode cue on v5 (no training) | — | 0.7658 vs 0.7688 | .662 vs .670 | | .645 | .840 vs .848 | | | | −.003 (other-fold selection mostly chose no cue) |

Recall: M1 COLLISION R@1/2/4/8 = .817/.868/.919/.950 vs M0 .759/.822/.891/.934; NEXAR COLLISION R@1 .704 vs .550.
BND2 w1 ENTRY R@1 .582 vs control .555. Boundary re-ranking of the top-K direct candidates (K 4/8, weight chosen on other folds):
±0.003 vs direct decoding. The boundary benefit comes from training the representation, not from decoding.

Parameter cost: risk +4.2k, boundary +0.3–0.5k, lane +0.1–0.5k, M1 +9.4k (72 extra input dims; residual extraction is CPU,
sparse Lucas–Kanade + RANSAC similarity on 160×90 frames, ~4 ms/frame, no RAFT).

### Lane pseudo-labels (quality gate before training)
* v1 (closest vehicle road-contact point vs ego-lane centre): **rejected**. AUC(score after vs before GT ENTRY) 0.49, rise +0.009,
  shifted-window null rises more; overlays show contact points vanish when the victim is close and lead vehicles dominate.
* v2 (fraction of the time-smoothed ego-lane corridor occupied by vehicle masks, lower 60 % of the image): **passed**.
  AUC 0.72, rise +0.21 vs shifted nulls +0.007 / +0.004, 79 % of clips rise, positive in every source (NEXAR +0.11), 99–100 % valid frames.

### Ensemble complementarity (finished part; B3 = v5 as built, CV 0.7688; add a 2-seed family, 4 disjoint draws)
| added family | CV all | NEXAR | >1000 | E / C acc | ENTRY / COLL oracle gain vs B3 | COLL error Spearman vs B3 |
|---|---|---|---|---|---|---|
| no-signal control A0 | 0.7629±.0013 | 0.647 | 0.644 | .643 / .833 | .057 / .031 | 0.41 |
| existing PH family (PH_repro) | 0.7651±.0064 | 0.642 | 0.635 | .649 / .832 | .045 / .027 | 0.40 |
| risk R3 | 0.7649±.0039 | 0.640 | 0.637 | .648 / .834 | .052 / .031 | 0.41 |

Adding any 5th family lowers v5's CV (v5's PH seed pair is a lucky draw, see `stage2_phase_supervision_study.md`). Risk behaves
exactly like adding another existing-family seed: no complementarity beyond ensemble size.

## Interim conclusions
1. **Largest standalone gain: residual motion as a learned input (M1)**, +0.023 over its matched M_motion control, CI excluding 0,
   4/5 folds, tight across seeds. Almost entirely COLLISION: +18.7 hits; long-clip COLLISION catastrophic 0.139 → 0.074;
   NEXAR 0.540 → 0.615. It complements M_motion rather than duplicating it (M2, residual alone, is already ≥ M0 on COLLISION).
   As a late decode cue it duplicates the camera-shift fusion (M3 negative). The network has to learn it.
2. **Boundary supervision with Δh (BND2 w1)**: +0.013 (CI excludes 0), improves ENTRY and COLLISION. It is the explicit
   transition signal, which supports "part of PH's benefit is boundary information". Re-ranking with it does not help.
3. **Lane intrusion (L2)**: +0.010, 4/5 folds, but improves COLLISION as much as ENTRY. **Not** a genuinely ENTRY-specific gain.
4. **Risk progression**: no standalone gain at 8 seeds (R3 −0.005) and no ensemble complementarity. It moves COLLISION metrics at
   3 seeds but that did not survive.
5. Still 3-seed for M1, BND2 and L2 (the phase study showed 3-seed winners shrink by ~0.003–0.005 at 8 seeds). Next steps if
   resumed: 8 seeds for M0/M1/BND2/L2 (jobs listed in `stage2/aux_signal_experiments/jobs.txt`, resumable with `queue.py`,
   which keeps ≥ 10 GB RAM free); v5 **replace** test (M_motion → M1) and add tests for BND2/L2 against same-size controls;
   combinations PH + BND2, BND2 + L2, and M1 + BND2.
