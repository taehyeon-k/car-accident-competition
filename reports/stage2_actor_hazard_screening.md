# Stage 2: fast screening of collision actor, dynamic labels, surprise, hazard ENTRY and actor ROI (2026-09-28)

Brief: `stage2/stage2_experiments.md`. **Everything here is single-seed fast-screening evidence.** Nothing below counts as an
improvement until it has been validated with multiple seeds and full CV. Single-seed noise is large: for E4_sa, seed 0 has native
ENTRY .585 while the 3-seed ensemble has .630. Differences of about ±.02 are noise.

## 1. Code and config changes

| File | Change |
|---|---|
| `stage2/actor/select_actor.py` | Collision-actor candidates and hand-set selector (superseded by the ranker); ByteTrack fragment linking backward (and forward up to COLLISION); gap interpolation ≤ 12 frames |
| `stage2/actor/ranker.py` | Learned actor selector: conditional logit, 14 label-free candidate features at the COLLISION estimate (window geometry / lane, growth, history intrusion rise). Fitted per fold with the Exp A oracle vehicle as target; each clip's actor comes from the ranker of the fold where it is held out |
| `stage2/actor/diag_actor.py` | Actor diagnostics plus the ORACLE change-point probe (diagnostic only) |
| `stage2/actor/actor_crops.py` | Actor crops per native frame: tight 1.25× and context 2× (224², full-resolution frame), DINOv3-S mean patch token |
| `stage2/actor/actor_features.py` | Actor slot features. Layouts: `a50` (D1), `a48p3` (D1b: actor + 3 prominence slots), `a147` (D5: 64-d PCA of tight + context crops), `a147c` (D5 crop-only). Caches `cache_actorfeat/<layout>/{train,val,eval}` |
| `stage2/actor/latent_predictor.py` | D3: per-fold MLP future-latent predictor (actor-crop and global-frame latents, horizons 1/2/4 steps, stride augmentation, training videos only). Surprise features appended (`a57s`, per-fold caches) |
| `stage2/aux_signal_experiments/model.py` | D4 hazard head `entry_hazard ∈ {replace, residual}`: h_t from pooled object/actor state plus hidden projection, no position input; log P(E=t) = log h_t + Σ_{j<t} log(1−h_j) |
| `stage2/aux_signal_experiments/losses.py` | `w_haz`: distribution loss on the hazard alone |
| `stage2/aux_signal_experiments/train.py` | `--entry-hazard`, `--w-haz`, `--dyn-label r,λ,late,warmup` (D2), `--obj-cache-val` (actor from a predicted COLLISION for validation clips) |
| `stage2/generalization/entry_suite.py`, `robust_eval.py` | Per-condition caches (`k1c50`, `k1c25`), `{fold}` caches, per-member caches (fusion of different branch inputs), ENTRY peak count, `--phys` lightweight physical decoder |
| `stage2/actor/{eval_arms.sh, eval_when_ready.sh, report.py, qual.py, timing.py}` | Evaluation, tables, examples, timing |

Configs of all arms: `stage2/aux_signal_experiments/jobs.txt` (run ids below). Base for every arm: ODS recipe (`--motion both
--stride-aug 0.5,0.25,0.25 --objmotion --obj-branch gate --w-obj-state 0.3`).

| ID | Brief | Arm | Delta |
|---|---|---|---|
| B0 | v8 | `E4_sa+E2_sa+XN4_sa@s0` | existing seed-0 members |
| B1 | ODS | `ODS@s0` | existing seed-0 members |
| D2 | dynamic labels | `D2o` | ODS + `--dyn-label 2,1.0,2.0,4` (r = 2 positions, 1 nat/position, late ×2, 4 warm-up epochs) |
| D4 | hazard | `D4Ro` / `D4Xo` | ODS + hazard residual (`w_haz` .5) / hazard replaces ENTRY |
| – | temporal-position GRL | `GRLo` | ODS + `--pos-grl 0.5` |
| D1 | collision actor | `D1` / `D1b` | actor slot only (a50, K 1) / actor + 3 prominence slots (K 4) |
| D3 | surprise | `D3` | D1 + 7 surprise features (per-fold predictor) |
| D5 | actor ROI | `D5` / `D5c` | actor slot with tight + context ROI latents / ROI only (no geometry / lane) |
| D4 on actor | | `D4a` / `D5H` | D1 + hazard residual / D5 + hazard residual |
| v15 | combination | `D4Ro+D5H` | fusion of two members (probability average; per-member caches) |
| v8 + v15 | | `E4_sa+E2_sa+XN4_sa+D4Ro+D5H` | vs matched control `E4_sa+E2_sa+XN4_sa+ODS@s0` |

## 2. Protocol

* Duplicate-clean 5-fold CV, 284 clips, seed 0 only. Each arm takes about 5 min to train (5 folds in parallel) and about 3 min to
  evaluate. Official 0.3 s rule.
* Conditions: native, ½, ⅓ rate (motion recomputed), and seeded crop50 / crop25 windows that contain ENTRY..COLLISION.
* Gap bins and bias pool the three rates.
* Actor at evaluation comes from the **v8 out-of-fold COLLISION prediction under the same condition**, so the actor is re-selected
  per crop window and per rate. During training, the actor comes from the GT COLLISION (training clips) or the E4_sa out-of-fold
  prediction (validation clips).
* Controls use the same protocol: B0 / B1 at seed 0. For 2-member fusions the control is `ODS@s01` (ODS seeds 0+1). For v8 + v15
  the control is v8 + ODS at seed 0.
* Caveats:
  * Tracks come from native-rate ByteTrack even under ½ and ⅓ (same assumption as OB_D).
  * Crop-window features use frames just before the window for their first derivatives.
  * The actor ranker for a clip was fitted on oracle labels of the other folds, including the evaluating fold's other
    validation clips. This is weak leakage (14 weights, only through the actor choice for training clips).
  * The crop and global PCAs were fitted label-free on all clips.

## 3–5. All methods: overall, per-gap ENTRY, crop / rate robustness

S = score; E = ENTRY; peaks = mean number of ENTRY modes (native); "vs B0/B1" = ENTRY rescue / harm (native + ⅓);
"actor ok/fail" = native ENTRY when the predicted actor holds the oracle vehicle at ENTRY (n 179) / not (n 105).

| arm | S nat/½/⅓ | ENTRY nat/⅓ | COLL | ENTRY <.5/.5–1/1–1.5/1.5–2.5/>2.5 s | bias | crop25 E (slope) | crop50 E (slope) | peaks | vs B0 | vs B1 | actor ok/fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** v8 s0 | .765/.749/.678 | .641/.493 | .852 | .90/.60/.53/.47/.16 | +.30 s | .514 (.52) | .609 (.74) | 1.73 | – | +62/−62 | .654/.619 |
| **B1** ODS s0 | .742/.715/.674 | .606/.528 | .831 | .91/.57/.52/.42/.22 | +.27 s | .486 (.46) | .609 (.69) | 1.75 | +62/−62 | – | .609/.600 |
| D2 dyn. labels | .738/.703/.668 | .613/.528 | .817 | .90/.59/.54/.36/.20 | +.30 s | .518 (.46) | .592 (.66) | 1.71 | +62/−60 | +46/−44 | .626/.590 |
| **D4 hazard residual** | .741/.712/.663 | **.616/.542** | .835 | .92/.62/.51/**.45**/.20 | **+.20 s** | **.532** (.47) | .592 (.71) | 1.90 | +68/−61 | +66/−59 | .626/.600 |
| D4 hazard alone (same run) | .739/.707/.659 | .613/.535 | .835 | .88/.62/.51/.45/.20 | +.23 s | .518 (.49) | .588 (.70) | 1.83 | +69/−65 | +63/−59 | .631/.581 |
| D4 hazard replace | .726/.724/.677 | .553/.521 | .828 | .88/.60/.51/.33/.16 | +.40 s | .465 (.47) | .567 (.68) | 1.55 | +62/−79 | +58/−75 | .536/.581 |
| position GRL | .716/.697/.663 | .546/.479 | .824 | .84/.58/.46/.28/.13 | +.30 s | .514 (.52) | .535 (.72) | 2.48 | +51/−82 | +40/−71 | .575/.495 |
| B1 + phys. decoder | .745/.714/.676 | .613/.532 | .831 | .90/.57/.55/.41/.20 | +.27 s | .482 (.47) | .595 (.70) | 1.75 | +62/−59 | +14/−11 | .603/.629 |
| D4 + phys. decoder | .741/.709/.666 | .620/.549 | .831 | .93/.65/.50/.43/.20 | +.20 s | .528 (.47) | .602 (.71) | 1.90 | +72/−62 | +65/−55 | .642/.581 |
| D1 actor only | .737/.729/.685 | .588/.542 | .838 | .90/.63/.51/.36/.20 | +.30 s | .472 (.43) | .574 (.71) | 1.92 | +65/−66 | +60/−61 | .570/.619 |
| D1 branch alone | .675/.687/.674 | .419/.503 | .831 | .81/.49/.43/.22/.13 | +.40 s | .521 (.45) | .468 (.69) | 2.81 | +76/−136 | +78/−138 | .475/.324 |
| D1b actor + 3 slots | .723/.713/.677 | .570/.532 | .824 | .92/.62/.49/.36/.18 | +.33 s | .479 (.51) | .574 (.71) | 2.04 | +58/−67 | +50/−59 | .564/.581 |
| D3 D1 + surprise | .727/.724/.674 | .592/.549 | .820 | .93/.58/.54/.40/.20 | +.30 s | .468 (.44) | .588 (.70) | 1.85 | +57/−55 | +61/−59 | .609/.562 |
| D5 actor ROI | .742/.723/.688 | .595/.560 | .835 | .93/.63/.51/.33/.22 | +.30 s | .503 (.48) | .606 (.71) | 1.85 | +70/−64 | +67/−61 | .626/.543 |
| D5 branch alone | .699/.688/.665 | .472/.479 | .835 | .80/.53/.40/.28/.16 | +.39 s | **.581** (.57) | .535 (.74) | 2.12 | +80/−132 | +78/−130 | .525/.381 |
| D5c ROI only | .730/.706/.671 | .599/.539 | .828 | .94/.63/.49/.33/.27 | +.33 s | .500 (.54) | .599 (.67) | 1.86 | +63/−62 | +65/−64 | .626/.552 |
| D5c branch alone | .678/.675/.669 | .465/.518 | .813 | .78/.56/.40/.30/.22 | +.30 s | .556 (.54) | .514 (.71) | 2.15 | +84/−127 | +83/−126 | .520/.371 |
| D4a actor + hazard | .732/.711/.691 | .581/.570 | .824 | .94/.62/.53/.34/.20 | +.30 s | .511 (.52) | .581 (.72) | 1.45 | +76/−71 | +61/−56 | .598/.552 |
| **D5H** ROI + hazard | **.750**/.724/.680 | **.616/.560** | **.845** | .90/.66/.49/.41/**.31** | +.30 s | .514 (.53) | .581 (.75) | 1.78 | +77/−65 | +75/−63 | .620/.610 |
| *control* ODS s0+s1 | .745/.734/.682 | .623/.546 | .824 | .93/.63/.53/.42/.18 | +.30 s | .521 (.54) | .620 (.74) | 1.84 | +61/−51 | +36/−26 | .615/.638 |
| D4 + D5 | .757/.744/.684 | .627/.556 | .845 | .92/.64/.53/.48/.22 | +.20 s | .539 (.52) | .616 (.73) | 1.91 | +67/−53 | +57/−43 | .648/.590 |
| **v15** = D4 + D5H | .746/.725/**.699** | .609/**.570** | .838 | .92/.65/.48/.47/.27 | +.27 s | .525 (.49) | .585 (.75) | 1.79 | +73/−60 | +60/−47 | .620/.590 |
| *control* v8 + ODS s0 | .767/.766/.700 | .665/.542 | .838 | .93/.64/.58/.51/.16 | +.27 s | .518 (.50) | .602 (.73) | 1.76 | +28/−7 | +58/−37 | .682/.638 |
| **v8 + v15** s0 | .764/.759/**.714** | .637/**.588** | **.859** | **.97/.68**/.54/.51/**.20** | +.27 s | **.546** (.55) | .599 (.74) | 1.85 | +52/−26 | +67/−41 | .642/.629 |

Readings:

* **v15 vs the 2-member control (ODS s0+s1).**
  * Better: ENTRY ⅓ +.024, 1.5–2.5 s +.05, >2.5 s +.09, overall ⅓ +.017.
  * Worse: native ENTRY −.014, crop50 −.035.
  * The simpler D4 + D5 fusion is better or equal on every ENTRY axis except ⅓ (.556 vs .570): native .627, 1.5–2.5 s .48,
    >2.5 s .22, bias +.20 s, crop25 .539, crop50 .616.
* **v8 + v15 vs v8 + ODS (seed 0).**
  * Better: ENTRY ⅓ +.046, overall ⅓ +.014, >2.5 s +.04, crop25 +.028, COLLISION +.021.
  * Worse: native ENTRY −.028 and native score −.003. Net ENTRY rescue vs B0 is similar (+26 vs +21).
  * Against the brief's bar (native, ⅓, 1.5–2.5 s and >2.5 s ENTRY all up without a crop25 regression), v15 is **not** a clear
    across-the-board win. It trades native ENTRY for frame-rate, very-long-gap and crop robustness.

## 6. Inference cost (this GPU, measured alone, JPEG decoding included; `stage2/actor/timing.py`)

| Component | Throughput | Per 150-frame clip (median) | Per 1297-frame clip (max) |
|---|---|---|---|
| RF-DETR Small + ByteTrack, every native frame (already needed by OB_D / ODS) | 168 frames/s, 0.96 GB | 0.89 s | 7.7 s |
| Ego-lane road head, every native frame (already needed) | 325 frames/s, 2.0 GB | 0.46 s | 4.0 s |
| OB_D / ODS crops, 3 per frame | 1452 crops/s | 0.31 s | 2.7 s |
| Actor crops, 2 per frame (tight + context; D5) | 1452 crops/s, 0.2 GB | 0.21 s | 1.8 s |
| Actor ranker + track linking (CPU) | – | ~0.1 s | ~0.8 s |
| Latent predictor (D3) | negligible | < 0.01 s | < 0.1 s |

* Detector calls per video equal the native frame count, and there is one tracker update per frame. Crop-encoder calls are
  3 × frames for ODS and 2 × actor-present frames for D5.
* The actor pipeline adds a dependency: v8 must predict COLLISION before the actor is selected, which means two passes. The
  second pass costs the heads only (milliseconds); the DINO features are reused.
* The actor front-end is cheaper than ODS's (2 crops vs 3 per frame). Shipping v15 still needs the whole detection, tracking and
  lane stack (about 1.5 s per median clip), which is what stopped v14.

## 7. Actor diagnostics (`stage2/actor/results/diag_actor.json`)

| | Hand-set selector | Learned ranker (used) |
|---|---|---|
| Exp A oracle ENTRY vehicle in the actor chain (n 249) | .554 | **.743** (GT COLLISION) / .719 (predicted) |
| Actor present at GT ENTRY | .873 | .961 / .940 |
| Identity continuity ENTRY→COLLISION | .847 | .919 / .896 |
| Predicted-COLLISION actor = GT-COLLISION actor | .849 | .905 |
| Median selection margin | .55 | 1.9 |

* With the hand-set selector the oracle vehicle was always present at COLLISION, but ranked 2nd or lower in 111 clips.
  Selection, not linking, was the problem.
* **ORACLE change-point probe (diagnostic only; GT COLLISION → actor → lane-intrusion crossing):** ENTRY hit .41 overall;
  .33 on 1.5–2.5 s and .13 on >2.5 s. Against v8 it rescues 23 clips and harms 89. The actor's trajectory has timing
  information, but a physical rule cannot read it out; it has to be learned.
* **Conditional on actor success,** native ENTRY for the best actor arm D5H is .620 when the actor holds the oracle vehicle at
  ENTRY and .610 when it doesn't. D5's is .626 vs .543, D5c's .626 vs .552. The ROI branch clearly depends on getting the actor
  right. Hazard (D5H) removes most of that dependence.
* 29 clips where D5H misses have the wrong actor, including `aihub_bb_1_190227_vehicle_126_194` (margin .21; the
  predicted-COLLISION actor differs from the GT one) and `aihub_bb_1_220521_vehicle_143_027` (margin .15). Low margin marks most
  selection failures, so it can serve as a reliability gate.

## 8. Qualitative examples (`stage2/actor/results/qual.txt`)

* **Successful long-gap correction:** `aihub_bb_1_211030_vehicle_208_55965` (gap 1.53 s). v8 + v15 error +0.20 s; B0 +0.93 s,
  B1 +0.33 s.
* **Failed actor selection:**
  * `aihub_bb_1_190227_vehicle_126_194`: margin .21, different actor for predicted vs GT COLLISION, error −1.60 s.
  * `aihub_bb_1_220521_vehicle_143_027`: margin .15, error +0.60 s.
  * Some misses have the right actor (`nexar_00025`, err −9.07 s): a very early false ENTRY, i.e. localisation failure after
    correct selection.
* **ENTRY pulled toward COLLISION:** among long-gap clips (≥ 1.5 s), misses predicted in the later half of ENTRY..COLLISION or
  after are 17/61 for B0 and B1 and 18/61 for v15 / D4 / D5H (15/61 for D5). **No method reduces this failure.** Examples:
  `aihub_bb_1_130420_vehicle_194_070` (err +1.13 s, gap 2.07 s) and `ccd_000475` (err +0.90 s, gap 1.80 s).
* **Dynamic-label shifts (D2):**
  * 3–16% of targets moved per epoch. At the last epoch almost all moves are −1 position; late moves are ≤ 2.9% (for example,
    fold 3 `aihub_bb_1_210621_vehicle_245_21554` +1).
  * Examples of −1 moves: `nexar_00324`, `mmau_002436`, `ccd_000246`.
  * No drift toward COLLISION; the shifts were consistently early rather than late, so the warning condition didn't occur.

## 9. Ranking of the five directions

| Direction | Verdict | Evidence |
|---|---|---|
| **D4 hazard ENTRY (residual)** | **Strong signal** (for a screen) | On ODS: ENTRY +.010/+.014, 1.5–2.5 s +.03, bias +.27 → +.20 s, crop25 +.046, net rescue +7 vs both baselines. On the actor ROI (D5H vs D5): native ENTRY +.021, >2.5 s +.09, COLLISION +.010. Replace mode is negative (−.053 native). Does **not** reduce multi-peak ENTRY (1.90 vs 1.75 modes); only replace mode does (1.55), at an accuracy cost |
| **D5 high-res actor ROI** | **Strong signal**, as a complementary branch | ⅓ ENTRY +.032 vs B1; ROI-alone crop25 .581 is the best of any model (v8 .514). Most of the signal is appearance (D5c ≈ D5). Hurts 1.5–2.5 s standalone (.33) unless combined with the hazard or with ODS |
| D1 collision actor (selection + identity) | Weak / uncertain as a replacement; useful as the enabler of D5 | Actor only: native ENTRY −.018, 1.5–2.5 s −.06. Adding the actor slot to the prominence slots (D1b) hurts. Selection is 74% oracle-consistent; wrong actors cost the ROI branch about 8 points |
| D3 future-latent surprise | Weak / uncertain | vs D1: ⅓ +.007, 1.5–2.5 s +.04 (about 2 clips), crop25 −.004 |
| D2 dynamic label assignment | Negative / neutral | ENTRY unchanged (.613/.528), 1.5–2.5 s −.06, COLLISION −.014. Label jitter is not the bottleneck |
| Position GRL (optional) | Negative | native ENTRY −.06, 1.5–2.5 s −.14 |
| Physical decoder | Weak / uncertain | +.004 to +.01 ENTRY, no gap-bin gain |

**Answer to the central question (screening level).** Treating ENTRY as a one-time hazard transition helps, and high-res evidence
on the collision actor adds frame-rate- and crop-robust ENTRY evidence that the global model lacks. Replacing the multi-object
representation with the single identified actor does not help: selection is imperfect (74%), and ODS's multi-object pooling holds
long-gap ENTRY better. Neither mechanism fixes the dominant long-gap failure: predicting ENTRY late, close to COLLISION, stays at
about 30% of long-gap clips. Local label flexibility and predictive surprise add little.

## 10. Smallest architecture worth rigorous validation next

1. **D5H as a new ensemble family:** actor ROI slot (tight + context crops) with a hazard-residual ENTRY head and ODS state loss;
   one object branch, K = 1. Plus **D4Ro** (ODS + hazard residual) as the second family.
2. Validation:
   * 3 seeds × 5 folds of D5H and D4Ro (about 30 min of GPU in total).
   * Compare v8 + D4Ro + D5H against v8 + ODS and v8 + D4Ro + D5 on the full robustness profile.
   * Run the two-factor leaderboard forecast on seeds 0–1, like earlier recipes.
3. Decision rule: adopt only if the multi-seed ENTRY at native rate does not fall below v8 + ODS while ⅓ and crop25 keep their
   gains.
4. Follow-up if it holds: a two-branch model (prominence slots and actor ROI as separate branches, with the actor branch gated by
   the ranker margin) in place of the probability-level fusion. Pulled-toward-COLLISION errors need a different mechanism, since
   none of the five directions moved them.
