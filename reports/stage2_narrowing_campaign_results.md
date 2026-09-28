# Stage 2: narrowing campaign results — why OB_D works and what limits it (2026-09-28)

Brief: `reports/stage2_narrowing_experiments.md`. Log: `stage2/generalization/results/entry_campaign.log`.
Code: `stage2/objtrack/diag_selection.py`, `diag_selector_recall.py`, `diag_lane_quality.py`, `objlane_features.py` (`--select heur`, K=6 cache),
`stage2/aux_signal_experiments/{model,losses,train}.py` (`--w-obj-state`, `--w-obj-rank`, `--obj-rgate`, `--obj-lane-drop`, `--pos-grl`),
`stage2/generalization/complementarity.py`, `robustness_profile.py` (recipes `v8+ODS`, `ODS+E2+XN4`),
`ens_cache.py` / `ens_search.py` (compact-v14 ensemble / head-TTA search; cache not yet built).

Protocol: duplicate-clean 5-fold CV (284 clips), official 0.3 s rule, v8 stride mix, no MM-AU/CCD extras. 3 seeds per arm unless marked
`[s0]` (1 seed, breadth-first screening). All arms share the OB_D base (`--motion both --stride-aug 0.5,0.25,0.25 --objmotion
--obj-branch gate`).

## Arms

| Arm | Change vs OB_D |
|---|---|
| SELH | causal relevance selector (SEL-H) over K=6 tracks, top 3 |
| SELS | K=6 tracks, no selection (all 6 to the branch) |
| SELHS | SEL-H top 4 of K=6 |
| ODR | + object ranking loss (`--w-obj-rank 0.5`) |
| **ODS** | + object BEFORE/ONSET/AFTER state loss (`--w-obj-state 0.3`) |
| ODRS | rank + state |
| OQ / OQD / OQG / OQDG | + 3 lane-reliability features; + lane dropout 0.3; + reliability gate; both |

## Results (E4 family level)

S = overall native / ½ / ⅓; E = ENTRY native / ⅓; gapE = ENTRY by gap bin (<0.5, 0.5–1, 1–1.5, 1.5–2.5, >2.5 s).

| Arm | S | E | gapE | 1.5–2.5 per seed | long-gap bias | crop25 E | crop50 E |
|---|---|---|---|---|---|---|---|
| E4_sa (v8 family) | .756/.736/.689 | .630/.503 | .94/.62/.51/.33/.22 | .29/.30/.32 | +.40 s | .514 | .599 |
| OB_D | .754/.739/.704 | .627/.553 | .94/.65/.52/.41/.20 | .40/.35/.34 | +.27 s | .532 | .606 |
| SELH | .754/.747/.702 | .620/.542 | .94/.66/.52/.36/.20 | .38/.35/.32 | +.47 s | .507 | .599 |
| SELS | .754/.745/.707 | .627/.553 | .93/.66/.51/.41/.22 | .40/.36/.33 | +.30 s | .500 | .567 |
| SELHS | .755/.744/.700 | .623/.535 | .94/.66/.54/.32/.18 | .28/.28/.36 | +.40 s | .542 | .606 |
| ODR | .746/.740/.696 | .620/.532 | .93/.67/.49/.36/.22 | .35/.33/.39 | +.40 s | .518 | .585 |
| **ODS** | .755/.742/.705 | **.641/.595** | .94/.67/.55/.42/**.31** | .42/.38/.33 | +.29 s | .525 | **.651** |
| [s0] OB_D@s0 (ref) | .726/.718/.675 | .592/.507 | .92/.58/.47/.40/.18 | .40 | +.27 s | .475 | .556 |
| [s0] ODRS | .728/.716/.677 | .585/.518 | .92/.58/.50/.35/.20 | .36 | +.30 s | .507 | .599 |
| [s0] OQ | .739/.706/.668 | .620/.507 | .94/.53/.51/.42/.20 | .42 | +.27 s | .514 | .585 |
| [s0] OQD | .735/.715/.673 | .581/.503 | .90/.56/.44/.45/.16 | .45 | +.27 s | .514 | .570 |
| [s0] OQG | .721/.712/.677 | .560/.546 | .92/.63/.49/.33/.16 | .33 | +.33 s | .518 | .578 |
| [s0] OQDG | .735/.717/.688 | .585/.574 | .90/.65/.53/.34/.11 | .34 | +.40 s | .482 | .585 |

Using the object head alone for ENTRY (`@entry_obj`) is much worse for every arm (ODS@entry_obj ENTRY .475/.500): the object branch works
as a residual correction on the frame model, not as a standalone detector.

### Ensembles

| Recipe | S | E | 1.5–2.5 | >2.5 | bias | crop25 E | crop50 E |
|---|---|---|---|---|---|---|---|
| v8 = E4_sa+E2_sa+XN4_sa | .764/.750/.710 | .641/.539 | .43 | .20 | +.33 s | .525 | .620 |
| OB_D+E2_sa+XN4_sa | .760/.748/.715 | .627/.549 | .42 | .18 | +.37 s | .525 | .609 |
| v8+OB_D | .766/.753/.709 | .641/.539 | .43 | .20 | +.33 s | .514 | .609 |
| **ODS+E2_sa+XN4_sa** | .765/.748/**.717** | .644/**.567** | **.48** | .22 | +.30 s | .511 | .630 |
| v8+ODS | **.772/.754**/.715 | **.655**/.567 | .46 | .22 | +.30 s | .521 | **.637** |

## Diagnostics

**Selector recall** (`diag_selector_recall.json`, 249 clips with an identifiable ENTRY vehicle): oracle vehicle in top-K

| Selector | @1 | @2 | @3 | @3, gap ≥ 1.5 s |
|---|---|---|---|---|
| PROM (area × centrality, OB_D) | .55 | .78 | .90 | .85 |
| SEL-H (causal relevance) | .74 | .90 | .95 | .96 |

OB_D ENTRY hit rate is .66 when the oracle vehicle is in PROM's top 3 and .44 when it is not (25 clips). SEL-H raises recall
substantially, but SELH/SELHS did **not** raise ENTRY. The 10 % of clips where PROM misses the vehicle cannot explain the plateau.

**Lane quality** (`diag_lane_quality.json`, split at ≥ 50 % defined rows and width CV < 0.3 around ENTRY): OB_D ENTRY .645 on good-lane
clips (200) vs .655 on poor-lane clips (84); E4 is .610 vs .655. OB_D's gain comes from good-lane clips, but poor-lane clips are not
failing worse than the frame model. The reliability features, gate and dropout (OQ*) were mixed or negative at 1 seed, so they were dropped.

**Complementarity with the v8 ensemble** (`complementarity.json`, rescue / harm counts; ENTRY hit):

* ODS alone vs v8: long-gap +21/−18, ⅓ +33/−17. Its misses are the least correlated with XN4_sa's (phi .56).
* ODS+E2+XN4 vs v8: long-gap +10/−3, >2.5 s +1/−0, native +9/−8, ⅓ +10/−2, crop25 +6/−10. By source: AIHUB +5/−2, CCD +0/−3,
  MMAU +2/−1, NEXAR +2/−2.
* v8+ODS vs v8: long-gap +8/−3, ⅓ +9/−1, crop25 +4/−5. Net ≥ 0 on every source (CCD +1/−1, NEXAR +1/−1).
* OB_D had almost no net rescue over v8. ODS is the first object arm with a positive net, concentrated at long gaps and at ⅓ rate.

## Bottleneck classification

Evidence supports **onset head/loss as the primary bottleneck**, with object selection secondary and lane quality minor:

1. **Onset head/loss (primary).** One change to supervision (ODS state loss) moved ⅓ ENTRY by +.042, >2.5 s by +.11, crop50 ENTRY by
   +.045 and 1.5–2.5 s by +.01, with the same tracks, crops and lanes. It is also the only change that made the branch complementary to v8.
2. **Candidate selection (secondary).** Recall matters per clip (.66 vs .44), but a +5 pt recall@3 gain gave no ENTRY gain. Selection
   combined with state supervision (SEL-H + ODS) has not been tested.
3. **Lane corridor (minor).** Poor-lane clips do not fail more often; reliability features and gating did not help.
4. **Detector/tracker, representation, position prior.** Not directly isolated. The Exp F position-probe GRL (`--pos-grl`) is
   implemented and smoke-tested but has not been run.

Negative results: ranking loss (ODR, ODRS), naive K=6 (SELS: crop50 −.04), lane reliability (OQ*).

## LB forecast and packaging

Two-factor forecast (score at ⅓ rate, crop25 ENTRY; LOO MAE .014; v8 forecast .5897 vs LB .5929):

| Recipe | Forecast |
|---|---|
| **ODS+E2+XN4** | **.603** |
| OBL3 | .594 |
| v8+ODS | .592 |
| v8 | .590 (LB .5929) |
| v13 | .589 |

* ODS full refits: `results/ODS_full/seed0-3` (all 349 clips, stop epoch 11). They are trained, but the ODS submission ("v14") needs the
  RF-DETR + ByteTrack + lane-head + crop pipeline at inference. The build was stopped because it is long and adds substantial inference
  time.
* A compact v14 (ensembles / head-level TTA over already-packaged non-object families; `ens_cache.py`, `ens_search.py`) was started and
  then stopped. No non-object recipe on record forecasts above v8.
* Submitted instead: `submit_U3G1PP-S1S3_v13S2_acc0p4.zip`, containing U3_G1PP stage 1/3 (accel threshold 0.4) and v13 stage 2.

## Next steps

1. Pursue onset supervision: object-specific boundary / change-point targets and state-loss weight and shape sweeps, plus SEL-H + ODS
   (selection and onset together).
2. Run the position-probe GRL (Exp F) on ODS to test the temporal-position prior hypothesis.
3. Make the object pipeline cheap enough to ship (a lighter detector, tracking at a lower rate, cached lane head) so that
   ODS+E2+XN4 (forecast .603) can be packaged.
