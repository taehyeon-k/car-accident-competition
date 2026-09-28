# Stage 2 night campaign (2026-09-28 23:12 KST -> 2026-09-29 08:00 KST)

Goal (user): > .85 validation without big robustness drops (hoped LB .65-.70). Constraints: inference cost near v8 (the detector
pipeline timed out the 60-min evaluation), no DACON view, never disturb the Stage 3 run.
Protocol: duplicate-clean 5-fold CV (284 clips), official 0.3 s rule; screening 1 seed per family vs the same seed of the baseline,
promising arms -> 3 seeds and full recipes; robustness = 1/2, 1/3 rate, crop50, crop25; LB two-factor forecast (seeds 0-1).

## Starting point
* v8 (3 seeds): .764 / .750 / .710 (1/1, 1/2, 1/3), forecast .590 (LB .5929). v8 with E4 distilled (KDe): .761 / .756 / .713, forecast .607.
* Ceiling check (115 existing prediction sets, native): best single ENTRY .697, COLLISION .859, side .944, evasion .732; combining the
  best of each component gives ~.80. .85 needs ENTRY around .78 -> the night must add ENTRY evidence, not re-mix models.

## Log
(Entries before 23:50 KST carry approximate times.)
* 23:20 — tracks started: (1) camera-compensated Farneback flow-grid features (stage2/night/flowgrid.py; 5 ms / frame);
  (2) stronger distillation: teacher = v8 + D5H ensemble (3 seeds each) + pre-collision expert, with side / evasion targets (kd2.py).
* ~23:3x-23:50 — event-specific family selection over cached OOF probabilities (stage2/night/ens_grid.py, seeds 0-1, 252 recipes of
  E4_sa/E2_sa/XN4_sa/KDe/KDe_E2/KDe_XN4): best forecast .626 (ENTRY = E4_sa+XN4_sa+KDe+KDe_XN4, COLLISION = distilled three),
  best native .770 (v8 at seeds 0-1: .749). Selection over 252 recipes on the same CV is optimistic -> not claimed yet.
  Per-source check: sampling spacing <= .126 s everywhere (< 0.3 s tolerance) -> zoom-in refinement dropped; AIHUB ENTRY misses are
  mostly near misses (74-78 % within .6 s, 45-48 % within .3 s, +.2 s late).
* ~23:3x-23:50 — KD weight sweep (seed 0): w 2 = .754/.711/.656, ENTRY .630/.472, crop25 .507 (vs w 1: .736/.709/.648, .623/.458, .549) —
  trade-off, not a win.
* ~23:3x-23:50 — seed-0 family screens vs E4_sa s0 (.728/.703/.653, ENTRY .585/.468) and KDe s0 (.736/.709/.648, .623/.458):
  FG (flow grid) .751/.734/.671, ENTRY .609/.489 — best at every rate; FGk .750/.723/.657; K2e (v8+D5H+PCE teacher) .745/.710/.661;
  K2ea (+attr KD) .747/.715/.656; AUX (actor-geometry aux targets) .742/.715/.676 with the best 1/3 ENTRY .521; KDe05 .737/.703/.657.
  -> flow features and aux actor targets are the new positives; stronger teacher / attribute KD not better.
* ~23:3x-23:50 — learned temporal stacker over 6 families' OOF distributions (stage2/night/stack.py, nested CV, all conditions as training
  examples): plain .767/.744/.711 ENTRY native .648 -> stacked .742/.728/.689 (.563); residual version (starts at the average)
  .743/.737/.688 (.563), COLLISION +.018. Negative for ENTRY (likelihood-optimal smoothing moves the argmax) -> dropped.
  AUXk (aux targets + KD) .740/.719/.657, ENTRY .606/.444, crop25 .556 — trade-off, not additive.
* ~23:3x-23:50 — FG component check (seed 0, native): evasion F1 .660 -> .729, side .883 -> .908, ENTRY .585 -> .609; AIHUB score .643 -> .707,
  NEXAR .618 -> .651. Flow is the first feature that moves evasion. Richer variant FG2 (4x6 grid + moving-pixel fractions, 125 d) queued.
* ~23:3x-23:50 — FGa (flow + aux) .742/.721/.690 (best 1/3 score), crop25 ENTRY .458; FGak (flow + aux + KD) .759/.715/.647 (best native
  single family), weak 1/3. Additions specialise families by condition -> test specialist ensembles.
* ~23:3x-23:50 — FG_E2 (E2 recipe + flow) .723/.708/.668 vs E2_sa s0 .730/.703/.649: better at 1/2-1/3, native ENTRY .570 vs .609 — mixed. Flow extraction for the 670 NEXAR extras running (flow-enabled XN4).
* ~23:3x-23:50 — seed-0 ensembles vs v8 s0 (.765/.749/.678, ENTRY .641/.493, crop25 .514): FG+FG_E2+XN4_sa .764/.747/.703 (ENTRY .627/.525,
  crop25 .525) -> 1/3 +.025; specialist mix FG+FGa+FGak+FG_E2+XN4_sa .770/.755/.695; + KDe .771/.753/.694. Native plateau ~.77.
  Queued 3-seed / forecast runs of FG+FG_E2+XN4_sa.
* 23:48 — 3-seed check: FG .759/.740/.692 (ENTRY .630/.518) vs E4_sa .756/.736/.689 (.630/.503): seed-0 gain mostly noise; evasion
  .691 vs .700 (seed-0 +.069 was luck). Only 1/3 ENTRY +.015 holds. Lesson: seed-0 screens swing +-.02 -> 3-seed confirmation for
  every candidate from now. Queued seeds 1-2 of K2e, FGk, AUX (+ v8-partner ensembles).
* 23:50 — window-mass (box-smoothed) decoding: neutral / negative (1/3 drops) -> dropped. Queued KDn (KD on native views only, 3 seeds): tests whether KD's 1/3 cost comes from the teacher's weaker 1/3 targets.
* 23:53 — label-limit check (5 diverse 3-seed recipes, native): in 33 clips (11.6 %) all agree within 0.2 s and all miss (median +0.61 s late, 82 % late; AIHUB 11, NEXAR 11, CCD 8, MMAU 3); 63 clips (22 %) missed by all five. -> ENTRY with today's evidence is capped near .78 even with a perfect selector; .85 overall needs ENTRY ~.78+.
* 23:54 — earlier-shifted ENTRY decoding (1 / 2 positions): crop25 ENTRY .549 -> .599 -> .627 but 1/3 ENTRY .525 -> .426 -> .243 (v8 E4 distilled); lateness is condition-dependent (a position prior in cropped windows), not a global offset -> dropped.
* 23:54 — queued KDc = KDe + window-crop augmentation (p .3), 3 seeds: targets the in-window late-ENTRY position prior (crop25 ENTRY is an LB-forecast factor).
* 23:54 — FG2_E2 (E2 + 4x6 flow grid) s0 .762/.718/.685 vs E2_sa s0 .730/.703/.649 (+.032/+.015/+.036) -> seeds 1-2 of FG2 / FG2_E2 queued with v8-partner ensembles.
* 00:01 — forecast (seeds 0-1): FG+FG_E2+XN4_sa .592 (1/3 .705, crop25 E .525) vs v8 .590, v8 E4 distilled .607 -> flow ensemble not better for the LB forecast. FG2 s0 .749/.714/.682 (crop25 .535, crop50 .634) awaiting 3 seeds.
* 00:02 — 3 seeds: FG+FG_E2+XN4_sa .764/.755/.707 (ENTRY .637/.528, crop25 .521) = v8 .764/.750/.710 -> flow neutral at ensemble level. FGh (flow + hazard, s0) .748/.734/.687 ~ FG.
* 00:04 — K2e (teacher = v8 + D5H + PCE, 3 seeds) .756/.729/.686, ENTRY .627/.489 < KDe (teacher = D5H + PCE) .769/.732/.694, .658/.514 -> adding frame-only v8 models to the teacher dilutes the object evidence; keep the object teacher.
* 00:04 — hypothesis: object-heavier teacher distils better -> teacher 3 = D5H + ODS + D4Ro (3 seeds each) + PCE .25 (cache_kd3); KD3 (E4 + teacher 3, 3 seeds) chained.
* 00:09 — FGk (flow + KD, 3 seeds) .772/.744/.699, ENTRY .665/.532, crop25 .570 — best single family (KDe .769/.732/.694; E4_sa .756/.736/.689). Flow and KD stack. Queued FGk_E2 / FGk_XN4 (3 seeds) and FGk ensembles (3 seeds + forecast).
* 00:11 — FGk+E2_sa+XN4_sa .764/.751/.704 and FGk+FG_E2+XN4_sa .763/.748/.714: ensemble ENTRY (.627-.630) below FGk alone (.665). Added suite --entry-fams (ENTRY from a family subset); queued ENTRY=FGk(+KDe) / COLLISION+attributes=all.
* 00:11 — AUX (actor-geometry aux targets, 3 seeds) .752/.741/.695, ENTRY .616/.528 vs E4_sa .756/.736/.689 (.630/.503): 1/3 ENTRY +.025, native -.004 — minor ingredient.
* 00:15-07:00 — results (3 seeds unless noted; forecast = seeds 0-1):
  | recipe | score nat / 1/2 / 1/3 | ENTRY nat / 1/3 | crop25 E | crop50 E | forecast |
  |---|---|---|---|---|---|
  | v8 | .764 / .750 / .710 | .641 / .539 | .525 | .620 | .590 (LB .5929) |
  | v8, E4 distilled (KDe; submitted 09-28) | .761 / .756 / .713 | .637 / .549 | .535 | .623 | .607 |
  | ENTRY = FGk, COLL + attr = FGk+E2_sa+XN4_sa | .777 / .750 / .706 | .665 / .532 | .570 | .630 | .614 |
  | ENTRY = FGk+KDe, COLL + attr = all | .778 / .754 / .702 | .665 / .535 | .556 | .613 | .607 |
  | FG2 + FG2_E2 + XN4_sa | .779 / .760 / .716 | .651 / .539 | .525 | .623 | .607 |
  | **FGk + FGk_E2 + FGk_XN4 (flow + KD, all three families)** | .773 / .741 / .709 | .669 / .521 | **.581** | **.641** | **.624** |
  Families (3 seeds): FGk .772/.744/.699; FGk_E2 .767/.744/.693; FGk_XN4 .774/.741/.675 (crop25 .588); FG2 .766/.737/.714;
  FG2_E2 .766/.727/.685; KD3 (object-only teacher) .762/.735/.692; KDn (KD native views only) .744/.728/.682 (worse: the stride-view
  targets help); KDc (KD + crops) .754/.734/.685 (crop aug does not add). FG_XN4 s0 .748/.708/.655; FG+FG_E2+FG_XN4 s0 .750/.747/.697.
* 07:00 — chosen: FGk + FGk_E2 + FGk_XN4 (flow grid + distillation in all three v8 families): best forecast .624 (+.034 vs v8, +.017
  vs the submitted v8 E4-distilled), native .773 (+.009), crop25 ENTRY .581 (+.056), crop50 .641; 1/3 .709 (= v8). Inference = v8
  + flow grid (~5 ms / frame on the 160 x 90 grey frames). Full refits (4 seeds, stop epochs 11 / 9 / 12) + package in progress.

## Conclusion
The 0.85 validation target was not reached: best 3-seed native score .779 (FG2 ensemble) / .777 (ENTRY-from-FGk), best forecast .624.
Evidence that .85 is out of reach with the current labelled data: 22 % of clips are missed by all five diverse strong recipes (ENTRY
ceiling ~.78 even with a perfect selector) and 11.6 % are consistent confident misses (~0.6 s late, all recipes agree) — a label
convention the models cannot learn from 349 clips; .85 needs ENTRY ~.78+ (best .669). What robustly helped: distillation from
object-aware teachers (object-only teacher; adding frame-only models dilutes it), camera-compensated flow-grid features combined with
distillation. Neutral / negative: flow alone (seed-0 gains were noise), actor-geometry aux targets (small 1/3 gain), stronger mixed
teacher, attribute KD, KD weight 2, KD on native views only, crop augmentation with KD, hazard head on flow, learned stacker,
window-mass / shifted decoding, attribute thresholds, free-space evasion features.
* 07:15 — LB facts (Stage-2-only): v8 .5929, v8+KDe .5886, v12 .5635, v13 .5609, v10 .5590. Unlabelled MM-AU/CCD pool = common factor of the losers. No CV view ranks the modern recipes: native / rates / crops / per-source / CCD-native, and the new test-format view (stage2/generalization/test50_view.py: ~10 fps stride, 50-frame window, COLLISION at frames 30-41 as in the DACON samples) ranks v12 .777 > v13 .764 > v10 .763 > v8kde .755 > v8 .753 — opposite to the LB. The test differs in content / labels, not format. Packaged v8 FGk (flow + KD in all three families) as a single-change LB test: submit_U3G1PP-S1S3_v8FGk-S2_acc0p4.zip (R2 submissions/2026-09-29).
