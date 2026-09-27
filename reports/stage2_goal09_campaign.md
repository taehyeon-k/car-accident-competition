# Stage 2 campaign toward CV > 0.9 / NEXAR > 0.8 (2026-09-27, in progress)

Starting point: best CV ensemble 0.7995 / NEXAR 0.721 (E4 + E2 + XN4, 8 seeds each); v8 recipe CV 0.787. Single-model E4_sa CV
0.739 (3 seeds). Judge: 5-fold CV over 349 clips, ≥ 3 seeds, paired against the control on the same folds/seeds (`analyze.py`).

## Where the points are lost (E4_sa CV, single models)
| source | score | ENTRY | COLLISION | side F1 | evasion F1 |
|---|---|---|---|---|---|
| AIHUB | 0.664 | 0.43 | 0.73 | 0.93 | 0.79 |
| CCD | 0.763 | 0.66 | 0.86 | 0.91 | 0.64 |
| MMAU | 0.868 | 0.82 | 0.95 | 0.93 | 0.72 |
| NEXAR | 0.638 | 0.45 | 0.74 | 0.86 | 0.61 |
ENTRY errors: 55 % within 0.3 s, 21 % at 0.3–0.6 s (near misses). Sampling grid spacing ≤ 0.13 s, so not quantisation.
AIHUB COLLISION misses are symmetric (12 % late / 15 % early); visual check (6 clips): side-swipes at the frame edge where the
contact frame is barely visible, one tunnel-entrance confusion — ambiguous contact, not a missing input.

## O1 — object tracks (RF-DETR Small + ByteTrack on every native frame; `stage2/objtrack/`)
* Hand rules on the true opponent (largest box at GT COLLISION): ENTRY hit ≤ 0.38 (corridor / area-growth / lateral-onset rules);
  opponent box at ENTRY = 4–93 % of its collision area (median 19 %). No crisp physical ENTRY event in the tracks.
* Learned: 44-d per-frame track features (top-4 road users: position, size, growth, approach, edge distance; stride-k versions)
  appended to the E4_sa input (`--objmotion --obj-cache stage2/objtrack/cache_objfeat`).
  CV 3 seeds: 0.735 vs 0.739 (**−0.004** [−0.015, +0.008], 2/5 folds; ENTRY −0.010). **LOSO 3 seeds: +0.024 mean, +0.021 worst,
  4/4 sources** (ens 0.703 / 0.606 → 0.716 / 0.622). Like H17: helps unseen sources, not in-domain.

## X1 — metadata-anchored expansion of the unlabelled pool (`stage2/generalization/unl_labels.py`, `train.py --extra-unl`)
* Metadata vs manual labels: MM-AU `t_co − 1` = COLLISION for 99 % (0.3 s); `t_ai + 5..7` = ENTRY for 66–69 %.
  CCD first-accident frame + 1 = COLLISION for 89 %. AIHUB label files have no timing.
* Labels per CV fold (offsets from the fold's train part; ENTRY = fold teacher argmax in a window anchored on the metadata).
  Pseudo-label accuracy on held-out labelled clips: MMAU C 0.99 / E 0.84, CCD C 0.89 / E 0.72, AIHUB (teacher only) C 0.78 / E 0.47.
* Results (E4_sa + extras, CV 3 seeds):
| arm | CV | NEXAR | ENTRY | COLLISION | >1000 C catastrophic | Δ vs E4_sa | folds | seed-ens |
|---|---|---|---|---|---|---|---|---|
| E4_sa | 0.739 | 0.638 | 0.597 | 0.822 | 0.058 | — | — | 0.768 |
| **XU_mc** (258 MMAU + 400 CCD) | **0.751** | 0.642 | 0.609 | **0.851** | **0.022** | **+0.013** [+0.000, +0.025] | 4/5 | 0.772 |
| XU_mca (+ 400 AIHUB, teacher-only) | 0.749 | 0.648 | 0.612 | 0.840 | 0.048 | +0.011 [−0.002, +0.023] | 3/5 | 0.764 |
  COLLISION is where the gain is (+0.029). AIHUB self-training adds nothing.

## X1 in the ensemble (goal_eval: v5 decoding, pooled 5-fold OOF, 3 seeds per family)
| ensemble (members / fold) | CV | NEXAR | ENTRY | COLLISION | side | evasion |
|---|---|---|---|---|---|---|
| v8 recipe E4_sa + E2_sa + XN4_sa (9) | 0.7874 | 0.7071 | 0.659 | 0.877 | 0.940 | 0.726 |
| all families with extras: XU_mc + E2_sa_xu + XN4_sa_xu (9) | 0.7892 | **0.7374** | 0.633 | 0.897 | 0.940 | 0.751 |
| E4_sa + E2_sa_xu + XN4_sa_xu (9) | **0.7973** | 0.7346 | 0.659 | 0.891 | 0.951 | 0.747 |
| all six families (18) | 0.7965 | 0.7371 | 0.656 | 0.897 | 0.948 | 0.738 |
| reference: v8 recipe, 5+5+3 seeds (15) | 0.7908 | 0.7106 | 0.668 | 0.883 | 0.934 | 0.721 |
Single families with extras vs their controls: E4 +0.013, E2 +0.012, XN4 +0.003 (XN4 already had extra NEXAR clips).
The 0.7973 row was picked among three 9-member combinations evaluated on the same CV (mild selection); the unpicked 18-member
union gives 0.7965, so the gain is not a selection artefact. At equal size the extras add ≈ +0.009 overall and **+0.028–0.030 NEXAR**
(best NEXAR so far; previous best 0.721 with 24 members). The gain is COLLISION (+0.014–0.020) and evasion; ENTRY is flat or lower.
Prior overall best 0.7995 used 24 members (8 seeds × 3 families, no stride aug); not yet compared at 24 members.

## Corrections (2026-09-27, after the independent review `stage2/STAGE2_EXPERIMENTS_DATASETS_LEADERBOARD_REVIEW_2026-09-27.md`)
* **Seed mismatch.** `analyze.py` compared 3-seed arms with the 6-seed E4_sa mean (0.7387); seeds 0–2 alone give 0.7421. Matched
  (`stage2/generalization/matched_cv.py`, seeds 0–2, source-stratified paired clip bootstrap): XU_mc **+0.0091** [−0.004, +0.024]
  (not +0.013), OT_sa −0.0075 [−0.019, +0.005] (not −0.004), H17 σ_E=2 −0.0123 [−0.026, +0.001]. E2_sa_xu +0.012 is unaffected
  (E2_sa had 3 seeds). The ensemble table above was already matched (explicit seeds 0–2 everywhere).
* **Oracle diagnostic.** `objtrack/diagnose.py` picks the opponent with GT COLLISION; its ≤ 0.38 hit rate does not show that no
  physical ENTRY cue exists. Only the learned box-statistics result is evidence (against repeating that representation).
* **Near-duplicates** (`stage2/generalization/dup_audit.py`, visually verified): 81 of 1,058 expansion clips duplicate a labelled
  clip (52 AIHUB, 19 MMAU, 10 CCD; 29 were in XU_mc's pool), and 19 labelled clips have a duplicate in another CV fold (AIHUB 9,
  CCD 5, MMAU 5; e.g. the same yellow-taxi crash as ccd_000721 and mmau_003028). Duplicate-clean re-scoring (65 validation clips
  dropped, `clean_eval.py`): v8 recipe 0.7743 / NEXAR 0.7071; all-xu 0.7759 / 0.7374; E4 + E2_xu + XN4_xu 0.7815 / 0.7303. The
  overall gain shrinks ~60 %; the NEXAR gain (+0.023–0.030; NEXAR has no duplicates) stands. XU_mc single-model clean: +0.0075.
  All earlier CV numbers in every report contain the 19 cross-fold duplicates.
