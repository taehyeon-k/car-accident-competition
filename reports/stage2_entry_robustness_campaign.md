# Stage 2 ENTRY robustness campaign (plan: `stage2/Stage2_Entry_Robustness_Experiments.md`, 2026-09-28)

All arms: E4 family (NT loss, global + residual motion, `stage2/aux_signal_experiments/train.py`), **v8 stride mix 0.5 / 0.25 / 0.25**
unless noted, AdamW 1e-3, OneCycle, batch 4, ≤ 30 epochs, patience 7, direct-decoder selection; 5-fold CV, seeds 0-2 (6 where
noted), probability-averaged ensembles, plain decoding (probability average + constrained anchors); **duplicate-clean** validation
(284 clips). Base parameter count 303,582 (E4) / 304,096 (E2). No MM-AU/CCD extras except in Experiment 5. FPS is never a model
input (only the metric, gap binning and the gap-OOD split definition use it). Code: `stage2/generalization/entry_suite.py`
(suite), `stage2/generalization/gapood/` (folds), `entry_campaign_eval.sh`; per-clip predictions `results/entry_suite/*.json`.

## Experiment 0 — diagnostic benchmark
Conditions: native, 1/2, 1/3 rate (motion recomputed), random 50 % / 25 % windows containing ENTRY..COLLISION. Gap bins pool the three
rates. Position slope = regression of predicted on true ENTRY position inside the window (1 = follows the evidence).
**Gap-OOD folds** (`gapood/`): L = train gap < 1.2 s (213 clips) → validate gap ≥ 1.5 s (73; all sources); S = train gap ≥ 0.8 s (219) →
validate gap < 0.5 s (71; 60 are MMAU → source-confounded). Near-duplicates of validation clips removed from training; fixed stop
epoch 9 (no selection on the OOD set). **Timing-prior baselines** (no video): PRIOR_POS (training-median relative ENTRY position) and
PRIOR_GAPR (reference COLLISION − training-median gap as a fraction of the clip / window).

| | ENTRY native / 1/3 | gap <.5 / .5-1 / 1-1.5 / 1.5-2.5 / >2.5 s | long-gap bias | crop25 ENTRY / slope |
|---|---|---|---|---|
| v8 recipe (E4+E2+XN4, 3 seeds) | .641 / .539 | .95 / .63 / .51 / .43 / .20 | +0.33 s | .525 / .58 |
| PRIOR_POS | .310 / .313 | .64 / .27 / .24 / .22 / .13 | +0.70 s | .292 / .04 |
| PRIOR_GAPR | .349 / .328 | .72 / .35 / .26 / .15 / .11 | +0.53 s | .296 / .06 |
The model is clearly evidence-driven overall (≈ 2x the priors, slope .58 vs .04). **But on gap-OOD-L the v8 E4 family finds only
.148 (native) / .213 (1/3) of long-gap ENTRY, biased +1.00 s**: without long-gap training examples, ENTRY collapses toward COLLISION.

## Results (E4 family unless noted; 3 seeds; E4_sa = v8's E4 family)
| arm | params | native / ½ / ⅓ | ENTRY nat / ⅓ | ENTRY gap <.5/.5-1/1-1.5/1.5-2.5/>2.5 | 1.5-2.5 per seed | long-gap bias | crop25 E / slope | crop50 E / slope | gap-OOD L nat / ⅓ (bias) |
|---|---|---|---|---|---|---|---|---|---|
| E4_sa (v8), 6 seeds | 303,582 | .753/.737/.687 | .623/.500 | .94/.62/.51/.32/.22 | .29 .30 .32 .41 .31 .30 | +0.40 | .500/.56 | .588/.72 | .148/.213 (+1.00) |
| **Exp 1** EB_f05 (level-0 [l, Δl] branch, α .5) | 320,223 | .717/.724/.687 | .539/.525 | .92/.62/.44/.30/.20 | .34 .27 .28 | +0.40 | .514/.53 | .567/.71 | |
| EB_f1 (α 1) | 320,223 | .725/.714/.689 | .553/.525 | .90/.67/.40/.31/.20 | .31 .25 .27 | +0.30 | .525/.51 | .556/.76 | |
| EB_g (gated) | 320,224 | .732/.720/.688 | .588/.525 | .94/.62/.47/.37/.16 | .39 .31 .26 | +0.30 | .514/.57 | .595/.72 | .180/.164 (+0.60) |
| EB_g branch alone | | .693/.697/.693 | .479/.539 | .89/.57/.42/.29/.13 | | +0.30 | .535/.53 | | |
| **Exp 2** OB_B (object appearance as input) | 334,782 | .749/.743/.693 | .627/.503 | .91/.65/.51/.36/.16 | .35 .27 .28 | +0.33 | .535/.56 | .581/.71 | |
| OB_C (object/lane branch, lane geometry only) | 309,824 | .747/.737/.696 | .599/.521 | .94/.60/.51/.40/.16 | .39 .28 .36 | +0.40 | .486/.46 | .595/.70 | .180/.180 (+0.87) |
| **OB_D (branch: appearance + lane geometry)**, 6 seeds | 311,936 | **.766/.750/.713** | **.648/.581** | .95/.66/.54/**.43**/.20 | **.40 .35 .34 .39 .37 .33** | **+0.27** | .528/.57 | .609/.71 | **.197/.311 (+0.90)** |
| OB_D branch alone | | .693/.689/.683 | .447/.482 | .80/.55/.37/.22/.09 | | +0.40 | .493/.47 | | |
| OB_E (OB_D + boundary branch) | 328,578 | .737/.733/.699 | .599/.553 | .92/.66/.48/.41/.27 | .34 .30 .33 | +0.40 | .528/.53 | .570/.73 | |
| **Exp 3** E4_sb (equal mix) | | .747/.727/.702 | .620/.553 | .96/.66/.52/.33/.24 | .33 .31 .39 | +0.40 | .468/.50 | | |
| v8 mix + full unl. consistency | | .740/.750/.703 | .588/.546 | .94/.65/.51/.30/.27 | .33 .25 .27 | +0.47 | .468/.47 | | |
| equal mix + full consistency (v13 E4) | | .740/.737/.715 | .592/.581 | .96/.65/.50/.35/.24 | .32 .23 .30 | +0.40 | .472/.48 | | |
| CS_v8C v8 mix + COLLISION-only consistency | | .746/.747/.696 | .609/.532 | .92/.62/.52/.38/.18 | .38 .30 .32 | +0.40 | .486/.47 | .574/.68 | .082/.230 (+0.95) |
| CS_v8W v8 mix + weak coarse ENTRY consistency (w .3, 8 bins) | | .754/.738/.711 | .620/.549 | .93/.62/.54/.36/.18 | .37 .31 .30 | +0.40 | .454/.49 | .574/.72 | |
| CS_sbC equal mix + COLLISION-only consistency | | .744/.738/.694 | .627/.525 | .94/.65/.51/.41/.16 | .38 .30 .36 | +0.30 | .500/.49 | .588/.71 | |
| **Exp 4** GB_v8 gap-balanced ENTRY loss | | .732/.714/.671 | .588/.500 | .95/.59/.44/.38/.18 | .36 .28 .33 | +0.33 | **.553/.60** | .595/.72 | |
| GA_v8 counterfactual gap warp (p .5) | | .757/.718/.695 | **.658**/.542 | .94/.65/.53/.36/.22 | .33 .29 .32 | +0.40 | **.556**/.52 | **.637**/.73 | .148/.246 (+0.90) |
| OB_D + gap warp | 311,936 | .754/.732/.697 | .641/.535 | .93/.64/.54/.38/.27 | .41 .36 .31 | +0.33 | .542/.52 | .613/.70 | .148/.230 (+0.93) |
| **Exp 6** HB_r local ENTRY ranking | | .753/.746/.691 | .630/.528 | .92/.65/.55/.35/.22 | .29 .28 .34 | +0.33 | .546/.56 | .616/.73 | |
| HB_rs ranking + BEFORE/ONSET/AFTER states | 304,353 | .754/.739/.695 | .641/.532 | .92/.65/.54/.38/.20 | .36 .37 .33 | +0.33 | .532/.53 | .637/.78 | .131/.328 (+0.80) |
| OB_D + gap warp + ranking | | .746/.733/.701 | .620/.563 | .92/.65/.56/.35/.22 | .36 .30 .36 | +0.40 | .528/.57 | .588/.71 | |
| **Exp 5 A** no extras (= E4_sa, v8 mix) | | .756/.736/.689 | .630/.503 | .94/.62/.51/.33/.22 | | +0.40 | .514/.56 | | |
| **Exp 5 B** + 629 MM-AU/CCD extras, COLLISION-only | | .730/.749/.712 | .570/.549 | .95/.64/.51/**.27**/.18 | .26 .28 .21 | **+0.47** | **.440/.41** | .546/.65 | |
| **Exp 5 C** + extras, COLLISION + teacher ENTRY | | .743/.755/.703 | .581/.532 | .95/.62/.48/.43/.22 | .40 .36 .31 | +0.33 | .454/.46 | .556/.69 | |
| E2_OB_D (E2 family + OB_D branch) | 312,450 | .772/.740/.696 | .676/.549 | .94/.68/.55/.38/.20 | .42 .30 .30 | +0.40 | .542/.57 | .641/.70 | |
| **OB_D + E2_OB_D + XN4_sa** (3 families) | | **.777/.763/.710** | **.662/.553** | .95/.67/.55/.43/.18 | .44 .36 .36 | +0.37 | **.539**/.58 | .634/.74 | |
| **v8 recipe** E4_sa + E2_sa + XN4_sa | | .764/.750/.710 | .641/.539 | .95/.63/.51/.43/.20 | .47 .35 .36 | +0.33 | .525/.58 | .620/.76 | |
Gap-OOD S (short-gap validation, MMAU-dominated) is ≈ .86-.96 native for every arm; not discriminative.

Source-wise ENTRY (native / ⅓ / crop25), v8 E4 vs OB_D: AIHUB .43/.45/.33 → .53/.45/.41; CCD .75/.32/.52 → .67/.44/.55; MMAU
.80/.78/.78 → .84/.84/.74; NEXAR .51/.44/.40 → .45/.45/.40. OB_D's gains are **not uniform across sources** (native CCD −.09, NEXAR −.06).

## Architecture notes
* **Exp 1 branch** (`AuxPyramid --entry-bnd`): level-0 features (first full-resolution ResidualDW block, before any avg-pooling) →
  [l_t, l_t − l_{t−1}] → Linear(256→64) + GELU + Dropout → Conv1d(64→1, k 3) → entry_bnd; entry = entry_v8 + α·entry_bnd (fixed α or
  α·sigmoid(g), g₀ = −2); branch-alone NT loss (w .5). +16.6k params. The existing BND2 head in E2 reads the final (already smoothed) h.
* **Exp 2 branch** (`--obj-branch gate`, `stage2/objtrack/`): per native frame the 3 most prominent detected vehicles (RF-DETR +
  ByteTrack; area × centrality); per vehicle 48 features: box geometry (8), ego-lane geometry at the contact point (7: corridor
  defined, signed intrusion depth past the lane boundary / half-width, lane overlap fraction, centre offset, their changes = ego-motion
  compensated lateral dynamics) and appearance (32-d PCA of a 224 x 224 DINOv3-S crop embedding + its change). Ego-lane corridor from
  the geometry model's road head on every native frame, ±4-frame median. Per object: Linear(96→32) (segment max + mean) → Conv1d(32,5)
  → Conv1d(1,3) → onset score; masked logsumexp over objects (no vehicle → 0) → gated residual on the ENTRY logits (+8.4k params).
  The object features reach **only** this branch; the trunk, COLLISION and attribute heads are v8's. Sanity: on the 73 long-gap
  clips the most-intruding vehicle's depth is .01 one second before ENTRY, .16 at ENTRY, .55 at COLLISION.
* **Exp 3** `--cons-entry-w / --cons-coll-w / --cons-entry-bins` on the (unlabelled) cross-rate consistency. **Exp 4** `--gap-aug`:
  the ENTRY..COLLISION interval is resampled on a virtual timeline (f ∈ LogU[0.6, 1.7]; real DINO frames, motion aggregated over the
  real frames each step spans) before adaptive sampling. **Exp 6** `--w-hard-rank` (logit(e) > logit(e±{2,4}) + .5, e+d before
  COLLISION) and `--w-hard-state` (BEFORE / ONSET / AFTER CE within ±8 positions on level-0 [l, Δl]; training only).

## Answers
1. **Why v8 transfers better than v10/v12.** v8 is the only recipe that is frame-rate robust *and* keeps truncation / position
   robustness. The extra MM-AU/CCD clips damage the latter **regardless of their ENTRY labels**: COLLISION-only extras (Exp 5 B) give
   the worst long-gap ENTRY (.27), latest bias (+.47 s) and worst crop ENTRY / slope (.440 / .41); teacher-ENTRY extras (C) recover
   long gaps (.43) but keep the truncation damage (.454 / .46). The extras are tight windows around the accident; their event positions
   teach a position prior. Earlier: frame-rate + crop-ENTRY features predicted v10 / v12 LB within ~0.005 when left out.
2. **Cause of the long-gap failure.** Mainly **training-gap priors**: removing long-gap training clips (gap-OOD-L) drops long-gap
   ENTRY from ~.43 to .15 with +1.0 s bias for every model; no loss design, pooling change or resolution change fixes this.
   Loss design (gap balance, ranking) and temporal pooling (Exp 1: level-0 branch) do not help long gaps. Spatial detail / object-lane
   evidence is the only lever that moves it (OB_D: .32 → .43 over 6 seeds; gap-OOD-L .148/.213 → .197/.311).
3. **Can ENTRY depend more on lane-intrusion geometry?** Partly. The lane-intrusion signal aligns with the labels, and the object +
   lane branch measurably reduces the late bias (+0.40 → +0.27 s) and the gap-OOD collapse — but only as a residual correction; the
   branch alone is a weak detector (.45 native ENTRY), and lane geometry without appearance hurts truncation.
4. **Long-gap ENTRY without losing frame-rate robustness?** Yes at family level: OB_D improves ⅓-rate (.687 → .713) and long gaps
   together. The consistency losses buy ⅓-rate robustness but cost truncation robustness in every variant; COLLISION-only consistency
   is the least harmful (Exp 3).
5. **Strongest evidence-backed candidate.** The 3-family **OB_D + E2_OB_D + XN4_sa** ensemble: native .777 (v8 .764), ½ .763 (.750),
   ⅓ .710 (.710), ENTRY .662 (.641), crop25 ENTRY .539 (.525), same slope, multi-seed. It does **not** meet all criteria: long-gap
   ENTRY equals v8's ensemble (.43, which already benefits from E2/XN4 diversity), long-gap bias is not below v8 (+0.37 vs +0.33),
   gains are not uniform across sources, and gap-OOD improves only modestly. Deploying it also requires RF-DETR + ByteTrack + crop
   encoder + road-head corridor inside the Stage 2 runtime. **Recommendation: no new leaderboard package yet** (plan rule); if one
   submission is spent to test the object/lane direction, this ensemble is the candidate. Otherwise keep v8.

## What genuinely improves visual grounding vs only CV
* Grounding (gap-OOD / bias / slope evidence): object + lane branch (OB_D); hard-boundary ranking (gap-OOD ⅓ .328, crop .546/.56);
  gap-balanced loss for truncation only (.553/.60).
* CV / native only: gap warp (native ENTRY .658, no gap-OOD gain); weak ENTRY consistency (⅓-rate only).
* Harmful: level-0 boundary branch (native ENTRY −.04…−.09); any MM-AU/CCD extras (truncation, long gaps); full ENTRY consistency.
