# Stage 2 phase-based temporal loss — ablation (2026-09-25)

Spec: `stage2/phase_loss.md`. Code: `stage2/long_context_v2_experiments/phase_loss.py` (loss + decoders), `models.py`
(`LCPyramid` phase head, `PhaseRefine` MS-TCN stage), `train.py --loss {p2,newtargets,phase} [--phase-refine 1]`,
`decode_phase.py`, `crop_shift.py`. Results: `results/{NT_newtargets,PH_phase,PH_phase_msrefine}/`, `results/phase_decoders.json`,
`results/crop_shift_*.json`, `results/fixed_eval_recipe_*.json`. Earlier runs untouched.

## Setup

* Same frozen geometry-DINOv3-S dense features, same adaptive sampling, same `LCPyramid` backbone (274k params), same optimiser/
  schedule/early stopping as the C0 baseline. Fixed 279/70 split with seeds 0–3; 5-fold CV over 349 clips (80 NEXAR) with CV seeds 0 and 1.
  All evaluation at batch 1, the submission setting.
* **Arms**
  * **C0**: original P2 loss (direct heads, normalized soft targets width 0.015, events .35/.35, attributes .15/.15, order .05).
  * **NT (control)**: new targets without phase terms: direct heads with Gaussian σ = 1 sampled position, 1.0·L_direct + 0.5·side + 0.5·evasion.
    It isolates the target/weight changes the spec makes besides the phase terms.
  * **PH**: the spec loss, 1.0 L_phase (weighted CE 1/3/1) + 0.75 L_transition + 0.25 L_direct + 0.05 L_mono + 0.5 side + 0.5 evasion.
  * **PH+MS**: the "suitable architecture": PH plus one MS-TCN refinement stage (phase probs → 4 dilated residual convs → phase
    logits; 291k params total). Every stage is supervised with the phase terms.
* Phase labels: PRE t < e*, BETWEEN e* ≤ t < c*, POST t ≥ c* at the sampled positions. Checked: no clip has e* = c* or e* = 0 at the
  sampled positions, so strict e < c decoding is valid. Guards exist anyway: skip transition loss if BETWEEN is empty; virtual PRE at t = −1.
  Everything is masked to valid (unpadded) positions; mono uses pairs where both t and t+1 are valid.
* **Pre-declared checkpoint selection:** PH/PH+MS early-stop with the structured decoder (phase sum + transition, weights 1/1); C0/NT with
  direct decoding. The other decoders are re-decoded offline from the *same* checkpoints and were never used for selection.

## Results (official offline score; fixed = mean ± std over 4 seeds; CV = pooled OOF, mean over 2 CV seeds)

| arm / decoder | fixed all | fixed NEXAR | fixed >1000 | CV all | CV NEXAR | CV >1000 | CV NEX ENTRY | CV NEX COLL | NEX E nMAE | NEX C nMAE | NEX E cat | NEX C cat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 / direct | 0.695±.023 | 0.465±.045 | 0.441 | 0.687 | 0.551 | 0.547 | 0.44 | 0.50 | .068 | .049 | .21 | .14 |
| NT / direct | 0.707±.005 | 0.454±.016 | 0.442 | 0.695 | **0.566** | **0.555** | 0.45 | **0.54** | .064 | .051 | .19 | **.13** |
| PH / direct | 0.721±.016 | 0.478±.021 | 0.471 | **0.709** | 0.557 | 0.545 | **0.46** | 0.51 | **.059** | .052 | .19 | .17 |
| PH / transition | 0.733±.011 | 0.489±.063 | 0.477 | 0.703 | 0.517 | 0.509 | 0.42 | 0.44 | .061 | .058 | **.16** | .18 |
| PH / struct phase | 0.729±.014 | 0.466±.055 | 0.452 | 0.683 | 0.452 | 0.445 | 0.33 | 0.34 | .103 | .108 | .26 | .28 |
| **PH / struct phase+tr (pre-declared)** | **0.741±.006** | **0.518±.041** | **0.508** | 0.686 | 0.460 | 0.452 | 0.35 | 0.35 | .096 | .100 | .24 | .27 |
| PH / struct phase+tr+direct | 0.738±.010 | 0.495±.039 | 0.483 | 0.688 | 0.462 | 0.454 | 0.36 | 0.35 | .096 | .100 | .25 | .27 |
| PH+MS / direct | 0.718±.010 | 0.477±.049 | 0.460 | 0.700 | 0.506 | 0.505 | 0.39 | 0.42 | .089 | .078 | .25 | .21 |
| PH+MS / struct phase+tr (pre-declared) | 0.695±.014 | 0.413±.039 | 0.391 | 0.675 | 0.469 | 0.467 | 0.31 | 0.39 | .106 | .105 | .26 | .29 |

### Does the structured decoder fix long-video wrong-event selections? No: on balance it adds them.

CV OOF, >1000-frame clips (77 clips × 2 CV seeds = 154 clip-evaluations); "catastrophic" = either event off by >10 % of the clip.

| model | catastrophic, direct | catastrophic, structured | fixed by structured | broken by structured |
|---|---|---|---|---|
| PH | 36 | 46 | 10 | 20 |
| PH+MS | 44 | 47 | 17 | 20 |

Examples (PH, CV seed 0; frames as (entry, collision)):
* FIXED `nexar_00381` GT (528, 553): direct (293, 555) → structured (484, 549); `nexar_00998` GT (582, 602): direct (463, 494) → structured (469, 503).
* FIXED (PH+MS) `nexar_00225` GT (593, 634): direct (279, 315) → structured (572, 610); `nexar_00328` GT (581, 606): direct (1077, 1094) → structured (605, 608).
* BROKEN `nexar_00225` GT (593, 634): direct (562, 581) → structured (0, 3); `nexar_00405` GT (550, 601): direct (573, 588) → (0, 3);
  `nexar_00103` GT (594, 616): direct (589, 620) → (19, 22); `nexar_00077` GT (494, 589): direct (560, 579) → (108, 111).

**Mechanism.** On held-out long clips the phase head often labels long *contiguous* blocks of pre-crash driving as POST. In
`nexar_00405` 90 % of the PRE region is predicted POST; `nexar_00225` has 65 %. Per-region argmax accuracy (PRE/BETWEEN/POST) is
0.75/0.46/0.85 on >1000-frame clips vs 0.90/0.74/0.94 on <300-frame clips. The structured score sums log-probabilities over ~150
positions per region, so one mislabelled block outweighs the sharp local transition/direct evidence and moves the pair toward
the clip start. Adding transition or direct terms (weights 1) doesn't counter an O(T) phase sum. The MS-TCN stage smooths the
phase sequence but does not make it more correct (its CV phase-model score is lower).

### Positional-artifact check (NEXAR collision ≈ 50 % of the clip)

CV fold models on their held-out NEXAR clips, cropped to 600 frames with the event midpoint at 30 / 50 / 70 % of the crop:

| model (decoder) | full | r = 0.3 | r = 0.5 | r = 0.7 |
|---|---|---|---|---|
| C0 (direct) | 0.551 | 0.571 | 0.553 | 0.580 |
| NT (direct) | 0.566 | 0.586 | 0.579 | 0.592 |
| PH (direct) | 0.557 | 0.578 | 0.566 | 0.556 |
| PH (structured, pre-declared) | 0.460 | 0.510 | 0.493 | 0.457 (coll. cat. 0.41) |

No arm collapses off-centre, so none relies on a learned "collision at mid-clip" rule. The structured decoder degrades most with the
event late in the window, which fits the POST-block failure.

## Effect of the loss

1. **Target/weight changes alone (NT vs C0)** give a small, lower-variance gain: CV all +0.008, NEXAR +0.015, fixed all +0.012.
2. **Phase terms as a training signal (PH/direct vs NT/direct)**: CV all **+0.014** (0.709, the best single-model CV aggregate so far),
   fixed all +0.014, NEXAR ±0 (CV 0.557 vs 0.566). The phase loss regularises the shared representation; it does not help long clips by itself.
3. **Structured decoding (PH/struct vs PH/direct)**: best fixed-split numbers (0.741 all, 0.518 NEXAR) but **clearly worse on the 80-clip
   CV** (NEXAR 0.460 vs 0.557; collision catastrophic 0.17 → 0.27). The 15-clip gain does not generalise.
4. **MS-TCN refinement**: worse than PH on every decoder.

## Best use found: phase-loss model as an ensemble member with motion fusion

Motion fusion = robust-z camera-shift score added to collision logits (β chosen on other folds in CV; frozen at 1.0 on the fixed
split) + native-frame snap; direct heads; ensemble-averaged probabilities including attributes; no position prior.

| ensemble (+ motion fusion + snap) | CV NEXAR (80) | CV all (349) | fixed NEXAR (15) | fixed all (70) | fixed >1000 |
|---|---|---|---|---|---|
| current P2 4-seed ensemble, plain decode | — | — | 0.500 | 0.757 | 0.492 |
| C0 + X_ema + M_motion | 0.643 | 0.759 | 0.557 | 0.743 | 0.528 |
| **C0 + X_ema + M_motion + PH** | **0.670** | **0.769** | **0.619** | **0.767** | **0.594** |

Adding the phase-loss member raises NEXAR ENTRY (CV 0.50 → 0.54; fixed 5/15 → 8/15) on both protocols. Caveat: PH was added after its
single-model results were seen, so the CV number carries some selection optimism; the fixed-split check was run with every setting frozen.

## Conclusions

* **Confirmed:** the phase loss is a useful auxiliary (best single-model CV aggregate, and it helps as an ensemble member). The
  structured whole-clip phase decoder is **harmful on long clips** because contiguous phase-labelling errors dominate a summed score;
  it adds more catastrophic wrong-event picks (20) than it fixes (10). MS-TCN refinement does not help.
* **Not supported:** that phase supervision by itself suppresses distant false peaks.
* **Next, if pursuing phase decoding:** normalise the phase term per region length (mean instead of sum), or use it only to re-rank
  the top-K direct/transition pairs (K ≈ 8) rather than searching all pairs; choose its weight by other-fold CV.
  **Recommended submission path:** C0 + X_ema + M_motion + PH seed ensembles refit on all 349 clips, direct decoding + motion fusion.
