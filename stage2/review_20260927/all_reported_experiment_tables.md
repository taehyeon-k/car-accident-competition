**Stage 2 documented experiment-table archive — snapshot 2026-09-27T10:29:08.960451+00:00**
Historical tables are preserved, including superseded claims. The main review explains protocol mismatches and later corrections. This is not a single comparable leaderboard.

**Source: [reports/stage2_complementary_signals_interim.md](/workspace/car-accident/reports/stage2_complementary_signals_interim.md)**


*Results (pooled 5-fold OOF, mean ± std over CV seeds; Δ = paired clip bootstrap vs matched control, 95% CI)*

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

*Ensemble complementarity (finished part; B3 = v5 as built, CV 0.7688; add a 2-seed family, 4 disjoint draws)*

| added family | CV all | NEXAR | >1000 | E / C acc | ENTRY / COLL oracle gain vs B3 | COLL error Spearman vs B3 |
|---|---|---|---|---|---|---|
| no-signal control A0 | 0.7629±.0013 | 0.647 | 0.644 | .643 / .833 | .057 / .031 | 0.41 |
| existing PH family (PH_repro) | 0.7651±.0064 | 0.642 | 0.635 | .649 / .832 | .045 / .027 | 0.40 |
| risk R3 | 0.7649±.0039 | 0.640 | 0.637 | .648 / .834 | .052 / .031 | 0.41 |

**Source: [reports/stage2_external_dataset_report.md](/workspace/car-accident/reports/stage2_external_dataset_report.md)**


*Outcome*

| Source | Total | High confidence | Borderline | Auto-reject |
|---|---:|---:|---:|---:|
| mmau | 11,730 | 806 | 2,198 | 8,726 |
| causalcrash | 273 | 8 | 98 | 167 |

**Source: [reports/stage2_generalization_research.md](/workspace/car-accident/reports/stage2_generalization_research.md)**


*Hypotheses tested*

| # | method | rationale | result |
|---|---|---|---|
| H1 | **temporal-rate (stride) augmentation** (each clip at 1/1, 1/2, 1/3 frame rate, p = .5/.25/.25; motion recomputed) | frame-rate invariance | **Robustness win**: 3-seed ens. stride 1/2/3 = 0.768 / 0.753 / 0.701 vs 0.749 / 0.708 / 0.633 (+0.019 / +0.045 / +0.068; stride-3 ENTRY 0.39 → 0.51); single-model CV +0.002; LOSO mean +0.002 (**CCD, the 10 fps source, +0.046**; MMAU −0.027) |
| H2 | per-clip input normalisation (DINO centring / motion median scaling / both) | domain offsets | **rejected**: LOSO mean −0.045 / −0.005 / −0.060, worst −0.10 (NEXAR); absolute appearance and motion magnitudes carry signal |
| H3 | source-balanced sampling | equal source exposure | **neutral**: LOSO mean +0.005, worst −0.020 (2/4 sources) |
| H4 | object-level independent motion (LK tracks not following the camera: grid counts, inward lateral / approaching velocity) | physical ENTRY cue | **rejected** (LOSO mean −0.002, worst −0.019; ENTRY not better) |
| H5 | cross-frame-rate consistency (binned-time KL, native view = teacher, λ .3) + H1 | temporal-corruption robustness (cf. arXiv 2403.20254) | **invariance ↑, accuracy ↓**: stride 1/2/3 = 0.751 / 0.739 / 0.714 (drop −0.037 vs −0.067 for H1) but native −0.017; CV −0.007 |
| H6 | mask MMAU ENTRY labels (convention test) | MMAU ENTRY labelled much closer to COLLISION | **rejected** (LOSO mean −0.014, worst −0.047): MMAU ENTRY labels help |
| H7 | pre-collision truncation augmentation (collision not visible, COLLISION loss masked) | break an "ENTRY = just before COLLISION" shortcut | **neutral** (CV −0.002; LOSO mean +0.009 / worst −0.009); long-gap ENTRY unchanged |
| H8 | causal ENTRY head (ENTRY from positions ≤ t + L only; leak-free, verified) | same shortcut, architecturally | **rejected** (L = 4: CV −0.014, CI excludes 0) and **still predicts ENTRY late on long gaps** |
| H10 | geometry-head features (ego vs alternative drivable area, corridor occupancy) under LOSO | physical evasion cue | **no evasion transfer** (held-out evasion F1 unchanged ≈ 0.55–0.65); LOSO mean +0.009 |
| H11 | per-source class-balanced side / evasion loss weights | label-shift (base-rate) shortcut | **modest consistent gain**: LOSO mean +0.010, 3/4 sources, worst −0.003; side F1 ↑ on all 4 held-out sources; evasion still ≈ chance |
| H12 | feature-space augmentation (10 % token drop + 0.1·σ Gaussian noise on frozen DINO tokens) | appearance robustness | **mixed**: LOSO mean +0.010 (AIHUB +0.036, CCD +0.052, MMAU −0.022, NEXAR −0.027) |
| H13 | EMA weights (0.99) | flat minima generalise | **best cross-source single-model result**: LOSO mean +0.022, worst +0.006, **4/4 sources**; but seed-ensemble mean −0.007 (less diversity) |
| H9 | semi-supervised cross-frame-rate consistency on 1,046 unlabelled AIHUB / CCD / MMAU clips (+ H1); LOSO excludes the held-out source's clips | more domains, no labels | **not a generalization win**: CV −0.009; LOSO mean +0.000 / worst −0.026 (CCD +0.061, AIHUB +0.013, MMAU −0.046, NEXAR −0.026); robustness 0.752 / 0.760 / 0.714 (flatter, like H5) |

*Recipe decision (ensemble level)*

| recipe (3 families × 3 seeds) | CV all / NEXAR (v5 decoding) | frame rate 1/1, 1/2, 1/3 (plain) | LOSO E4+E2 ensemble: single mean / worst | seed-ens mean / worst |
|---|---|---|---|---|
| v7 (no aug) | 0.784 / 0.717 | 0.772 / 0.720 / 0.648 | (E4 alone) 0.663 / 0.561 | 0.703 / 0.611 |
| **v8 = stride aug** | **0.787** / 0.707 | **0.775 / 0.759 / 0.718** | 0.672 / 0.564 | 0.711 / 0.606 |
| **v9 = stride aug + EMA** | 0.783 / 0.707 | 0.771 / 0.747 / 0.696 | **0.698 / 0.593** (4/4 sources ↑) | **0.720 / 0.613** |

*Confirmation with 6 seeds (final) — recommendation corrected*

| LOSO | single-model mean / worst (Δ vs E4) | ensemble mean / worst |
|---|---|---|
| E4 (6 seeds) | 0.659 / 0.543 | 0.699 / 0.610 |
| E4 + stride aug + EMA | +0.034 / +0.037, 4/4 sources | 0.715 / 0.585 |
| E2 + stride aug + EMA | **+0.042 / +0.064, 4/4 sources** | 0.722 / 0.618 |
| **E4+E2, stride aug, 12 members** | 0.673 / 0.570 | **0.721 / 0.617** |
| E4+E2, stride aug + EMA, 12 members | 0.697 / 0.593 | 0.712 / 0.600 |
| mixed EMA / non-EMA, 12 members | 0.685 / 0.578 | 0.714 / 0.607 |

**Source: [reports/stage2_geometry_dino_pretraining.md](/workspace/car-accident/reports/stage2_geometry_dino_pretraining.md)**


*Executive summary*

| Question | Answer |
|---|---|
| **Did geometry pretraining work?** | **Yes, modestly.** It trained stably. Held-out geometry improved on every task and every source. The learned features stayed close to the original DINOv3. |
| **Did it improve held-out geometry?** | **Yes.** Fresh heads trained on the *frozen* adapted backbone (same protocol as the original-backbone probe) score 0.837 vs 0.824 on the selection score. BDD human-labelled drivable mIoU: 0.756 → 0.796. Lane boundary-F1: 0.803 → 0.823. Curb bF1: 0.748 → 0.764. TuSimple lane bF1: 0.909 → 0.922. Flow EPE: 4.58 → 4.14 px. So the gain lives in the backbone, not only in the heads. Jointly trained Phase-1 heads add a little more (drivable 0.801, flow 3.74 px). |
| **Did it preserve the original representation?** | **Yes.** With the anchor loss: mean patch cosine to the original 0.989, linear CKA 0.992, Gram correlation 0.995, CLS cosine 0.982. Without the anchor: 0.977 / 0.985 / 0.992 / 0.964. PCA feature maps are visually unchanged. |
| **Did it improve downstream Stage-2 performance?** | **Slightly, and only clearly for the no-anchor variant.** Identical small temporal probe, 5-fold video-level CV × 3 seeds, competition-style score. Original 0.565. Adapted + anchor 0.572 (paired Δ +0.006, t≈1.1, not significant). Adapted without anchor **0.583 (Δ +0.018, t≈2.3, 4/5 folds better)**. On the clean fixed split (validation videos never seen in pretraining): 0.530 → 0.556 (anchor) and **0.585** (no anchor). The gain comes from collision timing (+2.5 to +4.5 pts) and evasion-space F1 (+5 pts). Entry timing and entry side do not improve; they drop slightly, within noise. |
| **Which geometry tasks helped most?** | **Temporal correspondence (flow) is the task that matters for Stage 2.** Static-only adaptation scores −0.004 vs the original (entry timing −3.7 pts). Adding flow gives +0.012 over static-only (t≈1.9). On held-out geometry, the largest backbone gains: drivable/ego-lane segmentation (+4 mIoU pts on BDD human labels, +7–8 pts vs teacher on accident footage), flow/correspondence (−10% EPE), vanishing point (−9%). Objects/contact/depth gained less (+1–2 pts). Single-task removal at 4k steps changed other tasks by ≤0.005 (no measurable cross-task transfer). |
| **Which dataset sources helped most?** | Each source mainly helps its own domain. BDD100K carries most road/object quality. TuSimple lifts TuSimple lanes/VP. **Local accident + BATON footage is the only source that improves accident-domain objects (+3 mIoU) and dashcam flow (EPE 3.2 → 2.6 px on accident, 3.9 → 2.5 on BATON)** ([Dataset contribution](#dataset-contribution)). |
| **Is the checkpoint worth using for Stage 2?** | **Yes, as a drop-in replacement to try, using the no-anchor Phase-1 backbone.** It improves the controlled Stage-2 probe on both protocols and keeps CKA 0.985 with the original. Expect a small gain (≈+0.02 score in the probe), not a step change. Confirm inside the real Stage-2 model before committing. |

*1.1 Sources, subsets and footprint*

| Source | What was downloaded / used | Split unit | Train | Held-out val | Disk |
|---|---|---|---|---|---|
| **BDD100K 100K images** | Official mirror `dl.yf.io/bdd100k`: `bdd100k_images_100k.zip` (5.7 GB, deleted after preprocessing), `bdd100k_labels.zip` (per-image JSON: lane polylines incl. road curb, drivable/alternative areas, 2-D boxes; 181 MB), `bdd100k_drivable_maps.zip` (451 MB). No video. | official train/val (each image = a distinct video) | 70,000 images | 10,000 images (3,000 used for final eval, 1,500 for model selection) | 4.9 GB images @448×800 + 0.6 GB zips |
| **TuSimple** | Full dataset only exists as a 23 GB Kaggle zip. I read its central directory over HTTP and fetched **only frames 12, 17, 19, 20** of every clip (`scripts/remote_zip.py`), 3.0 GB. | clip | 3,626 clips (frame 20 labelled) | 600 official *test* clips with `test_label.json` | 2.9 GB raw + 0.4 GB processed |
| **Local accident footage** (CCD / Nexar / AI-Hub) | Already on disk (Stage-2 decoded frames). **Only frames were used, never the 4 Stage-2 labels.** | video | 201 Stage-2 *train* videos → 3,783 anchor frames | 50 Stage-2 *val* videos → 948 frames | 0.5 GB |
| **BATON dashcam** (Stage-3 data, 41 comma.ai routes, 20 fps) | Already on disk. | route (hash split) | 31 routes → 4,402 frames | 10 routes → 1,301 frames | 0.2 GB |
| KITTI | **Not downloaded** (see recommendation). | – | – | – | 0 |

*1.2 Label provenance (human vs pseudo vs derived)*

| Target | BDD100K | TuSimple | Accident / BATON |
|---|---|---|---|
| Drivable 3-class (bg / ego-direct / alternative) | **human** (drivable maps) | **derived** ego lane only (from human lanes; other pixels ignored) | **pseudo** (road teacher, conf ≥ 0.7, else ignored) |
| Lane markings | **human** (parallel lane polylines; crosswalks and stop lines ignored) | **human** | **pseudo** |
| Road curb / edge | **human** (`lane/road curb`) | ignored | **pseudo** |
| Objects (6-class: bg, car, truck, bus, two-wheeler, pedestrian) | **pseudo masks from human boxes** (SAM 2.1 box prompts) | pseudo (RF-DETR boxes → SAM) | pseudo (RF-DETR boxes → SAM) |
| Vehicle–road contact band | **derived** from accepted masks + human drivable map | derived | derived |
| Relative depth | pseudo (DA-V2) | pseudo | pseudo |
| Vanishing point / horizon | **derived** from human lane lines (7,246 BDD imgs) | derived (3,576 train clips) | none |
| Optical flow | – | pseudo (SEA-RAFT) | pseudo (SEA-RAFT) |

*1.3 Pseudo-label acceptance*

| | BDD100K | TuSimple | Accident | BATON |
|---|---|---|---|---|
| SAM instance masks accepted (geometric consistency with the prompt box; rejects become ignore regions) | 84.6% (729,651 / 862,143 train; tiny human boxes < 6 px are rejected by design) | 97.8% | 95.8% (17,427 / 18,183 train) | 98.6% (12,155 / 12,324) |
| Vehicle contact targets accepted | 232,975 of 879,040 vehicle instances (26%). Rejected: too small 25%, off-road/occluded below 25%, mask rejected 14%, truncated 9% | 19,952 / 32,477 | 9,374 / 21,364 | 6,773 / 15,399 |
| Flow pairs accepted (fw/bw-consistent fraction ≥ 0.3) | – | 99.2% (10,792 / 10,878) | 94.0% (10,667 / 11,349) | 90.2% (11,914 / 13,206) |
| TuSimple ego-lane derivable | – | 3,545 / 3,626 train; 578 / 600 val | – | – |

*2. Teachers*

| Task | Model / checkpoint | Confidence strategy | Qualitative assessment |
|---|---|---|---|
| Instance masks | **SAM 2.1 Hiera-Small** (`facebook/sam2.1-hiera-small`), box-prompted | Accept only if SAM IoU score ≥ 0.70 (human boxes) / 0.80 (detector boxes), mask tight-box IoU with the prompt box ≥ 0.7, and fill ≥ 25%. Otherwise the box region is *ignored*. | Very clean on cars/buses; weaker on thin two-wheelers and tiny far objects (those are ignored). ![sam](stage2_geometry_assets/pseudo_sam_depth.jpg) |
| Boxes (non-BDD) | **RF-DETR Small**, COCO (project checkpoint, sha256 d81979a9…), score ≥ 0.5, COCO→compact taxonomy | score threshold | Reliable for vehicles; misses heavily blurred crash frames. |
| Depth | **Depth Anything V2 Small** (local, Apache-2.0), relative inverse depth, 518×924 | Flip test-time augmentation: affine-align both passes, confidence = exp(−rel. disagreement/0.25), pixels < 0.3 ignored | Good relative ordering; object boundaries and thin poles are unstable (low confidence). DA-V2-S rather than B/L: already local and Apache-licensed. |
| Optical flow | **SEA-RAFT-S** (spring-S config, project checkpoint), run at 288×512 and stored at stride 4 (fp16) | Mixture-Laplace uncertainty × UnFlow forward/backward consistency × in-view test | Good on vehicles and ego-motion. Near-camera road leaving the frame is masked by the in-view test, as intended. ![flow](stage2_geometry_assets/pseudo_flow.jpg) |
| Road / lane / curb (accident + BATON only) | **DINOv3 ViT-B/16** (official local checkpoint), last 4 blocks + head trained 7k steps on **BDD human labels only** (`configs/road_teacher_vitb.yaml`). BDD val: drivable mIoU 0.839, ego-lane IoU 0.841, lane bF1 0.881, curb bF1 0.820. | Flip-TTA; drivable softmax conf < 0.7 ignored, lane/curb probability in (0.3, 0.7) ignored | Conservative: night, rain and camera-pointing-at-sky frames are mostly ignored. ![road](stage2_geometry_assets/pseudo_road.jpg) |
| Camera (VP) | No learned teacher. Derived from human lane lines (median of consistent pairwise intersections). | Kept only if ≥ 50% of intersections lie within 2.5% of the image width of the median | Precise where available (only ~10% of BDD images). |

*5. Training*

| | Phase 0 (frozen probe) | Phase 1 + anchor | Phase 1 no anchor | Probe on frozen adapted |
|---|---|---|---|---|
| GPU | L40S 46 GB (shared) | same | same | same |
| Steps | 6,000 | 10,000 | 10,000 | 6,000 |
| Batch per step | 12 static images + 4 pairs (8 images) | same | same | same |
| Effective batch | 20 images (no accumulation) | same | same | same |
| LR heads / backbone | 5e-4 / – | 1e-4 / 5e-6 | 1e-4 / 5e-6 | 5e-4 / – |
| Optimizer / schedule | AdamW (wd 1e-4 heads, 0.05 backbone), 300–500 warmup, cosine to 5% | ← | ← | ← |
| Init | fresh heads | Phase 0 heads (LP-FT) | Phase 0 heads | fresh heads |
| Precision | bf16 autocast | ← | ← | ← |
| Wall time | 24.7 min | 90 min | 88 min | 72 min (contended) |
| Samples seen | 120k images | 200k images | 200k images | 120k images |
| Peak VRAM | 2.2 GB | 4.5 GB | 4.3 GB | 2.2 GB |
| Best step | 6000 (last) | 10000 (last) | 10000 (last) | 6000 (last) |

*6. Held-out geometry results*

| Metric | Frozen original + heads (Phase 0) | **Frozen ADAPTED + fresh heads** | Phase 1 joint (+anchor) | Phase 1 joint (no anchor) |
|---|---|---|---|---|
| BDD drivable mIoU (3-cls, human) ↑ | 0.756 | 0.796 | 0.801 | **0.803** |
| BDD ego/direct-lane IoU (human) ↑ | 0.761 | 0.797 | 0.801 | **0.802** |
| BDD lane boundary-F1 ±4 px (human) ↑ | 0.803 | 0.823 | **0.826** | 0.826 |
| BDD curb boundary-F1 (human) ↑ | 0.748 | 0.764 | 0.770 | **0.772** |
| TuSimple lane boundary-F1 (human) ↑ | 0.909 | 0.922 | **0.922** | 0.922 |
| BDD object mIoU (SAM @ human boxes) ↑ | 0.676 | 0.684 | **0.699** | 0.699 |
| BDD vehicle IoU ↑ | 0.867 | 0.870 | **0.875** | 0.875 |
| BDD contact boundary-F1 (derived) ↑ | 0.893 | 0.900 | **0.905** | 0.905 |
| BDD contact row error, px @448 ↓ | 1.97 | 1.91 | 1.85 | **1.84** |
| BDD depth δ<1.25 vs teacher ↑ | 0.893 | 0.905 | **0.913** | 0.913 |
| BDD depth AbsRel vs teacher ↓ | 0.099 | 0.092 | **0.086** | 0.086 |
| BDD depth ordinal accuracy ↑ | 0.979 | 0.981 | **0.983** | 0.983 |
| BDD VP error, px @448×800 (derived) ↓ | 27.8 | 25.4 | 23.7 | **23.3** |
| BDD horizon error, px ↓ | 9.29 | 9.29 | 9.05 | **8.93** |
| TuSimple VP error, px ↓ | 8.97 | 7.98 | 7.70 | **7.60** |
| Accident drivable mIoU vs teacher ↑ | 0.828 | 0.902 | 0.906 | **0.911** |
| Accident lane bF1 vs teacher ↑ | 0.865 | 0.889 | 0.897 | **0.898** |
| Accident object mIoU vs teacher ↑ | 0.544 | 0.553 | **0.558** | 0.555 |
| Accident contact bF1 ↑ | 0.856 | 0.860 | 0.864 | **0.865** |
| Accident depth δ1 vs teacher ↑ | 0.836 | 0.848 | **0.854** | 0.854 |
| BATON drivable mIoU vs teacher ↑ | 0.922 | 0.960 | 0.966 | **0.967** |
| TuSimple flow EPE, px @448 ↓ | 4.58 | 4.14 | 3.74 | **3.74** |
| TuSimple flow EPE, Δ=400 ms ↓ | 6.39 | 6.26 | **5.83** | 5.86 |
| Accident flow EPE ↓ | 2.67 | 2.59 | **2.49** | 2.49 |
| BATON flow EPE ↓ | 2.59 | 2.50 | 2.37 | **2.37** |
| **Geometry selection score** ↑ | 0.824 | 0.837 | 0.844 | **0.845** |

*7. Representation drift*

| | + anchor | no anchor |
|---|---|---|
| Mean patch cosine (adapted vs original) | **0.989** | 0.977 |
| Worst-image patch cosine | 0.962 | 0.905 |
| CLS cosine | 0.982 | 0.964 |
| Patch-token norm, original → adapted | 7.78 → 7.75 | 7.78 → 7.75 |
| Gram (patch-relation) mean abs diff | 0.017 | 0.033 |
| Gram correlation | 0.995 | 0.992 |
| Linear CKA (patch features) | **0.992** | 0.985 |
| Patch cosine by source (BDD / TuSimple / accident / BATON) | 0.990 / 0.987 / 0.991 / 0.987 | 0.980 / 0.976 / 0.979 / 0.973 |

*8. Stage-2 probe: original vs geometry-adapted (controlled)*

| Backbone | Score | Entry acc@0.3 s | Collision acc@0.3 s | Side macro-F1 | Evasion macro-F1 |
|---|---|---|---|---|---|
| A. original frozen | 0.530 ± 0.014 | 0.240 ± 0.043 | 0.673 ± 0.025 | **0.859** ± 0.016 | 0.542 ± 0.099 |
| B. adapted (+anchor) | 0.556 ± 0.012 | 0.287 ± 0.034 | 0.680 ± 0.016 | 0.818 ± 0.046 | 0.631 ± 0.063 |
| B'. adapted (no anchor) | **0.585** ± 0.008 | **0.293** ± 0.025 | **0.733** ± 0.009 | 0.799 ± 0.061 | **0.706** ± 0.068 |

*8. Stage-2 probe: original vs geometry-adapted (controlled)*

| Backbone | Score | Entry acc@0.3 s | Collision acc@0.3 s | Side macro-F1 | Evasion macro-F1 |
|---|---|---|---|---|---|
| A. original frozen | 0.565 ± 0.032 | **0.381** ± 0.049 | 0.620 ± 0.065 | **0.858** ± 0.040 | 0.574 ± 0.082 |
| B. adapted (+anchor) | 0.572 ± 0.023 | 0.361 ± 0.036 | 0.645 ± 0.052 | 0.840 ± 0.039 | 0.622 ± 0.095 |
| B'. adapted (no anchor) | **0.583** ± 0.020 | 0.369 ± 0.038 | **0.665** ± 0.050 | 0.848 ± 0.037 | **0.626** ± 0.065 |

*8. Stage-2 probe: original vs geometry-adapted (controlled)*

| | Score | Entry | Collision | Side F1 | Evasion F1 | Per-fold score Δ |
|---|---|---|---|---|---|---|
| B − A | +0.006 ± 0.006 (t 1.1, 11/15 wins) | −0.020 ± 0.014 | **+0.025 ± 0.009** | −0.018 ± 0.011 | **+0.048 ± 0.012** | −0.008, +0.003, +0.029, −0.000, +0.007 |
| B' − A | **+0.018 ± 0.008** (t 2.3, 10/15 wins) | −0.012 ± 0.017 | **+0.045 ± 0.009** | −0.010 ± 0.013 | **+0.052 ± 0.016** | −0.008, +0.027, +0.050, +0.017, +0.004 |

*Dataset contribution*

| Metric | BDD only | BDD + TuSimple | + accident + BATON (full-4k) |
|---|---|---|---|
| BDD drivable mIoU (human) ↑ | **0.786** | 0.783 | 0.781 |
| BDD lane bF1 (human) ↑ | **0.819** | 0.816 | 0.817 |
| BDD curb bF1 (human) ↑ | 0.762 | **0.762** | 0.761 |
| **TuSimple lane bF1 (human)** ↑ | 0.898 | **0.919** | 0.915 |
| **TuSimple VP error px** ↓ | 10.3 | **7.6** | 8.1 |
| BDD object mIoU ↑ | **0.692** | 0.689 | 0.689 |
| **Accident object mIoU vs teacher** ↑ | 0.521 | 0.523 | **0.551** |
| Accident depth δ1 ↑ | 0.841 | 0.838 | **0.845** |
| TuSimple flow EPE px ↓ | – | **3.75** | 4.06 |
| **Accident flow EPE px** ↓ | – | 3.20 | **2.58** |
| **BATON flow EPE px** ↓ | – | 3.85 | **2.46** |

*Ablations*

| Held-out metric (4k steps) | full-4k | no flow (static only) | no depth | no road |
|---|---|---|---|---|
| BDD drivable mIoU (human) ↑ | 0.781 | **0.784** | 0.782 | – |
| BDD lane bF1 (human) ↑ | **0.817** | 0.816 | 0.816 | – |
| BDD curb bF1 (human) ↑ | **0.761** | 0.760 | 0.759 | – |
| TuSimple lane bF1 ↑ | 0.915 | **0.916** | 0.915 | – |
| BDD object mIoU ↑ | 0.689 | **0.690** | 0.688 | 0.689 |
| BDD contact bF1 ↑ | 0.900 | **0.900** | 0.900 | 0.899 |
| BDD depth δ1 ↑ | 0.903 | 0.904 | – | **0.904** |
| BDD VP error px ↓ | **25.7** | 25.7 | 26.0 | 25.9 |
| TuSimple flow EPE ↓ | 4.06 | – | 4.04 | **4.00** |
| Accident flow EPE ↓ | 2.58 | – | 2.57 | **2.55** |
| BATON flow EPE ↓ | 2.46 | – | 2.45 | **2.42** |

*Ablations*

| Backbone | CV score | Entry acc | Collision acc | Side F1 | Evasion F1 | Fixed-split score |
|---|---|---|---|---|---|---|
| original | 0.565 ± 0.032 | **0.381** | 0.620 | **0.858** | 0.574 | 0.530 |
| adapted, **static only** (no flow) | 0.562 ± 0.029 | 0.344 | 0.637 | 0.856 | 0.599 | 0.535 |
| adapted, static + **temporal** (full-4k) | **0.573** ± 0.020 | 0.365 | **0.652** | 0.834 | **0.615** | **0.549** |

*Storage analysis*

| Item | Size |
|---|---|
| BDD100K (processed 448×800 images 4.9 GB + label zips 0.6 GB; 5.7 GB image zip deleted after use) | 5.5 GB |
| TuSimple (selected raw frames 2.9 GB + processed 0.4 GB) | 3.3 GB |
| Accident + BATON processed frames | 0.7 GB |
| Pseudo-label cache (all sources, compressed npz, fp16 flow, uint16 depth, uint8 masks) | 9.6 GB (BDD 4.5, TuSimple 1.2, accident 1.5, BATON 1.8) |
| Stage-2 probe features (5 backbones × 2.2 GB) | 11 GB |
| Checkpoints + logs + visualizations (`outputs/geometry_pretrain`) | ≈5.4 GB (road teacher 0.8 GB; 4 main runs + 6 ablations 4.6 GB) |
| New pretrained assets (DINOv3 ViT-S 165 MB, SAM 2.1-S 176 MB) | 0.3 GB |
| **Total new** | ≈ 40 GB |
| Peak temporary usage | ≈ +6 GB (BDD image zip during preparation) |
| Free space: start → now | 74 GB → 35G (the drop is shared with concurrent Stage-3 caches/runs; the letterboxed-frame cache (3.9 GB) and the road teacher's optimizer checkpoint (1 GB) were deleted at the end) |

**Source: [reports/stage2_goal09_campaign.md](/workspace/car-accident/reports/stage2_goal09_campaign.md)**


*Where the points are lost (E4_sa CV, single models)*

| source | score | ENTRY | COLLISION | side F1 | evasion F1 |
|---|---|---|---|---|---|
| AIHUB | 0.664 | 0.43 | 0.73 | 0.93 | 0.79 |
| CCD | 0.763 | 0.66 | 0.86 | 0.91 | 0.64 |
| MMAU | 0.868 | 0.82 | 0.95 | 0.93 | 0.72 |
| NEXAR | 0.638 | 0.45 | 0.74 | 0.86 | 0.61 |

*X1 — metadata-anchored expansion of the unlabelled pool (`stage2/generalization/unl_labels.py`, `train.py --extra-unl`)*

| arm | CV | NEXAR | ENTRY | COLLISION | >1000 C catastrophic | Δ vs E4_sa | folds | seed-ens |
|---|---|---|---|---|---|---|---|---|
| E4_sa | 0.739 | 0.638 | 0.597 | 0.822 | 0.058 | — | — | 0.768 |
| **XU_mc** (258 MMAU + 400 CCD) | **0.751** | 0.642 | 0.609 | **0.851** | **0.022** | **+0.013** [+0.000, +0.025] | 4/5 | 0.772 |
| XU_mca (+ 400 AIHUB, teacher-only) | 0.749 | 0.648 | 0.612 | 0.840 | 0.048 | +0.011 [−0.002, +0.023] | 3/5 | 0.764 |

**Source: [reports/stage2_goal_campaign.md](/workspace/car-accident/reports/stage2_goal_campaign.md)**


*Result*

| ensemble (equal weights, plain average) | CV all | CV NEXAR | CV >1000 | fixed all | fixed NEXAR |
|---|---|---|---|---|---|
| v5 as submitted (C0 + X_ema + M_motion + PH) | 0.769 | 0.670 | 0.660 | 0.764 | 0.619 |
| **E4 + E2 + XN4** (best) | **0.7995** | **0.721** | 0.709 | 0.784 | 0.689 |
| E4 + E2 + XN4 + XN4e0 | 0.7989 | 0.716 | 0.704 | 0.782 | **0.706** |
| E4 + E2 + XN4 + M1 + v5 families | 0.7908 | 0.709 | 0.692 | **0.788** | 0.697 |

*Families (single-model CV; 8 seeds unless noted; control = matching recipe without the change)*

| family | recipe | CV all | NEXAR | E acc | C acc | note |
|---|---|---|---|---|---|---|
| A0 (control) | NT loss, no motion | 0.699 | 0.551 | .571 | .745 | |
| **E4** | NT + global & residual motion input | **0.740** | 0.642 | .608 | .823 | +0.04 over A0; residual motion is the main gain |
| **E2** | E4 + boundary heads [h, Δh] | 0.743 | 0.640 | .608 | .826 | ≈ E4 alone, better in ensembles |
| **XN4** | E4 + 670 unlabelled NEXAR clips (COLLISION from public `time_of_event`, anchored ENTRY pseudo-labels, weight 0.5) | 0.743 | **0.657** | .592 | **.839** | >1000 COLLISION catastrophic 0.071 → **0.013** |
| XN2 (3 seeds) | E2 + NEXAR expansion | 0.742 | 0.676 | .593 | .835 | worse ENTRY; lowers the ensemble |
| XN4e0 (3 seeds) | expansion, COLLISION-only (ENTRY masked) | 0.739 | 0.654 | .605 | .823 | no standalone gain |
| M1 (3 seeds) | P2 loss + both motion | 0.730 | 0.615 | .589 | .813 | |

*Tried and rejected (3-seed CV, matched controls)*

| idea | result |
|---|---|
| geometry-head spatial features (lane/vehicle grids, corridor occupancy; 83-d per frame) | +0.005 alone, −0.005 with motion; ENTRY unchanged |
| 14×25 high-resolution frozen tokens (vs 7×10) | **−0.026** (0/5 folds, overfits) |
| ENTRY loss weight 2 | −0.008 |
| sharper targets σ = 0.5 positions | −0.015 (0/5 folds) |
| event-window attribute stacker (logistic regression on geometry/motion windows) | side 0.937 → 0.896, evasion 0.716 → 0.667 |
| per-attribute family / threshold selection on other folds | −0.003 (unstable choices) |
| event-specific family subsets (ENTRY / COLLISION chosen on other folds) | +0.002 with 3-seed families, noisy with 8 |
| residual motion as a late decode cue on v5 | −0.003 |
| decode-time ENTRY/COLLISION bias shift | no systematic bias exists (median signed error ≈ 0) |

**Source: [reports/stage2_iterative_search.md](/workspace/car-accident/reports/stage2_iterative_search.md)**


*Current evidence*

| Run | Change from P2 | Offline score | ENTRY | COLLISION | Decision |
|---|---|---:|---:|---:|---|
| B0 | Existing ASFormer, fixed 128 | 0.689 | 0.557 | 0.814 | Baseline; best long COLLISION |
| P2 seed 0 | Four-level compact pyramid, adaptive K | 0.730 | 0.629 | 0.786 | Best single seed so far |
| P5 global coarse | Global attention at L3 | 0.696 | 0.614 | 0.743 | Reject: worse |
| P6 long oversample | All train clips once + extra >500-frame draws | 0.677 | 0.600 | 0.714 | Reject: worse |
| P7 event categories | Event-conditioned side/evasion heads | 0.710 | 0.643 | 0.771 | Reject as whole model: attributes fell |
| A1 ASFormer adaptive | Original larger temporal head, adaptive K | 0.535 | 0.371 | 0.543 | Reject: severe overfit |
| P2 seed 1 | Independent initialization/shuffle | 0.703 | 0.571 | 0.743 | Variation confirmed |
| P2 seed 2 | Independent initialization/shuffle | 0.688 | 0.543 | 0.771 | Variation confirmed |
| P2 seed 3 | Independent initialization/shuffle | 0.729 | 0.586 | 0.800 | Variation confirmed |
| ENS_P2_seed0123 | Equal probability average of four P2 seeds | **0.752** | 0.629 | 0.829 | Best aggregate so far |
| P8 hard negative | +0.1 ranking penalty on distant false peaks | 0.693 | 0.600 | 0.757 | Reject: no event gain |
| P9 adaptive target | Cap dense target width at two sampled positions | 0.689 | 0.586 | 0.771 | Reject: no gain |
| P10 EMA 0.99 seed 0 | EMA-smoothed checkpoint selection | 0.741 | 0.671 | 0.786 | Promising single-head gain; replication underway |
| GAP_ENS_P2_0123 | Train-fitted sampled-index gap prior on four-seed ensemble | 0.742 | 0.600 | 0.829 | Reject: loses ENTRY hits |

**Source: [reports/stage2_leaderboard_vs_nexar.md](/workspace/car-accident/reports/stage2_leaderboard_vs_nexar.md)**


*Data*

| Subset | n | v1 E3 (LB 0.441) | v3 proxy P2 ens (LB 0.462) |
|---|---:|---|---|
| ALL | 70 | 0.689 [0.625, 0.750] | 0.752 [0.691, 0.811] |
| AIHub | 18 | 0.663 [0.516, 0.791] | 0.774 [0.646, 0.880] |
| CCD | 17 | 0.696 [0.583, 0.799] | 0.766 [0.679, 0.838] |
| MM-AU | 20 | 0.815 [0.729, 0.892] | 0.898 [0.827, 0.954] |
| **NEXAR** | 15 | **0.522 [0.382, 0.650]** | **0.500 [0.360, 0.639]** |
| non-NEXAR | 55 | 0.734 [0.665, 0.795] | 0.820 [0.764, 0.869] |

*Data*

| Subset | Δ val | 95% CI | P(Δ ≥ +0.021) |
|---|---:|---|---:|
| NEXAR (15) | −0.022 | [−0.143, +0.104] | 0.25 |
| non-NEXAR (55) | +0.086 | [+0.017, +0.151] | 0.97 |
| ALL (70) | +0.063 | [+0.002, +0.122] | 0.91 |

*Update 2026-09-26: new leaderboard results*

| Submission (Stage 2) | offline fixed all / NEXAR (15) | CV all / NEXAR (80) | LB S2 |
|---|---|---|---|
| v1 E3 ASFormer | 0.689 / 0.522 | — | 0.4410 |
| v3 P2 4-seed refit (proxy: 279-clip ensemble) | 0.757 / 0.500 | — | 0.4618 |
| NEXAR specialist (3 heads + NEXAR KDE prior, every clip) | 0.670 / 0.744 | — | 0.437 |
| length-gated hybrid (>500 frames → 9-head specialist, else P2) | 0.802 / 0.710 | — | 0.4154 |
| **v5 LC-v2 15-head ensemble + motion fusion** | 0.767 / 0.619 | 0.769 / 0.670 | **0.5314** |

**Source: [reports/stage2_long_context_v2_report.md](/workspace/car-accident/reports/stage2_long_context_v2_report.md)**


*1. Verified prior claims (batch-1 re-evaluation, fixed val)*

| Model | all | NEXAR | non-NEXAR | >1000 |
|---|---|---|---|---|
| E3 fixed-128 ASFormer | 0.689 | 0.522 (E .33 / C .67) | 0.734 | 0.492 |
| P2 seed0 | 0.735 | 0.473 (E .47 / C .33) | 0.804 | 0.463 |
| P10 EMA seed0 | 0.738 | 0.481 | 0.807 | 0.467 |
| P2 4-seed ensemble | **0.757** | 0.500 (E .33 / C .47) | 0.826 | 0.492 |
| P5 global coarse / P6 long oversample / P8 hard neg / P9 adaptive target | 0.69 / 0.68 / 0.69 / 0.69 | 0.37 / 0.45 / 0.35 / 0.44 | | |

*A. Controlled NEXAR context length (15 val clips; crops keep ENTRY..COLLISION and their relative position)*

| context | score | ENTRY | COLL | ENTRY MAE (fr) | COLL MAE (fr) | E norm | C norm | side F1 | eva F1 | E cat | C cat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.576 | .60 | .40 | 16.0 | 18.2 | .126 | .143 | .87 | .64 | .33 | .53 |
| 256 | 0.553 | .53 | .40 | 28.3 | 30.6 | .111 | .120 | .87 | .64 | .40 | .20 |
| 512 | 0.585 | .53 | .47 | 39.3 | 39.5 | .077 | .077 | .87 | .70 | .13 | .20 |
| 768 | 0.571 | .53 | .40 | 42.4 | 48.1 | .055 | .063 | .87 | .76 | .13 | .13 |
| 1024 | 0.548 | .47 | .40 | 52.5 | 46.3 | .051 | .046 | .87 | .76 | .20 | .13 |
| full (~1200) | 0.500 | .33 | .47 | 67.5 | 59.3 | .056 | .049 | .87 | .60 | .20 | .13 |

*B. Candidate Recall@K (70 val clips; NMS radius = 1.5 % of positions; hit = official 0.3 s)*

| model | NEXAR ENTRY @1/2/4/8/16 | NEXAR COLL @1/2/4/8/16 | overall ENTRY @1/8 | overall COLL @1/8 |
|---|---|---|---|---|
| E3 | .33/.53/.60/.60/.67 | .60/.67/.67/.73/.80 | .54/.83 | .80/.94 |
| P2 seed0 | .40/.47/.73/.73/.73 | .40/.47/.60/.67/.73 | .61/.93 | .80/.91 |
| P10 | .40/.53/.60/.67/.67 | .47/.53/.73/.80/.87 | .64/.91 | .80/.96 |
| P2 ens4 | .27/.67/.80/.80/.80 | .47/.53/.73/.73/.73 | .61/.94 | .83/.94 |

*3. Phase 2/3 — architectures (all use the same frozen dense features, 279/70 split, P2 recipe)*

| run | params | fixed all | fixed NEXAR | fixed non-NEX | CV all | CV NEXAR | CV NEX E / C | CV NEX C-cat |
|---|---|---|---|---|---|---|---|---|
| C0 avg pool (control) | 274k | 0.695±.023 | 0.465±.045 | 0.757±.019 | 0.687±.002 | **0.551±.025** | .44 / .50 | .14 |
| C1 max pool | 274k | 0.710±.018 | 0.491±.048 | 0.769±.016 | 0.689 | 0.497 | .40 / .40 | .15 |
| C2 SGP strided DW | 275k | 0.700±.021 | 0.533±.057 | 0.745±.011 | 0.699 | 0.497 | .34 / .49 | .16 |
| D phase λ=0.1 | 274k | 0.719±.010 | 0.467±.030 | 0.787±.005 | 0.700 | 0.482 | .36 / .45 | .16 |
| D phase λ=0.25 | 274k | 0.712±.027 | 0.515±.040 | 0.765±.023 | 0.689 | 0.528 | .43 / .44 | .14 |
| D phase λ=0.5 | 274k | 0.696±.010 | 0.440±.052 | 0.764±.016 | 0.694 | 0.521 | .39 / .51 | .16 |
| F1 coarse bi-SSM (2 coarsest levels) | 291k | 0.710±.021 | 0.460±.038 | 0.777±.019 | 0.685 | 0.503 | .38 / .46 | .15 |
| F2 coarse local attention | 407k | 0.711±.014 | 0.469±.057 | 0.776±.014 | 0.708 | 0.550 | .48 / .49 | .14 |
| X context-crop aug 0.5 | 274k | 0.735±.018 | 0.517±.043 | 0.794±.018 | 0.693±.004 | 0.527±.010 | .39 / .51 | .14 |
| X EMA 0.99 | 274k | 0.706±.010 | 0.445±.019 | 0.776±.008 | 0.702±.007 | 0.540±.013 | .39 / .52 | .15 |
| X crop 0.5 + EMA | 274k | **0.740±.023** | 0.492±.063 | **0.807±.018** | 0.695±.002 | 0.531±.014 | .43 / .49 | .13 |

*3. Phase 2/3 — architectures (all use the same frozen dense features, 279/70 split, P2 recipe)*

| ensemble | fixed all | fixed NEXAR | fixed non-NEX | CV all | CV NEXAR | CV >1000 |
|---|---|---|---|---|---|---|
| original P2 ens4 (reference, fixed only) | 0.757 | 0.500 | 0.826 | — | — | — |
| C0 control | 0.744 | 0.511 | 0.807 | 0.702 | 0.548 | 0.543 |
| X crop | **0.764** | 0.555 | 0.820 | 0.714 | 0.542 | 0.527 |
| X EMA | 0.736 | 0.479 | 0.806 | 0.714 | **0.565** | **0.560** |
| X crop + EMA | 0.757 | 0.542 | 0.815 | 0.714 | 0.559 | 0.552 |

*E. Top-K pair scorer (K = 8 × 8, out-of-fold candidates from an inner 5-fold split; fixed epochs; seed 0)*

| variant | fixed all | NEXAR | non-NEX | notes |
|---|---|---|---|---|
| generator (C0 seed0) | 0.696 | 0.456 | 0.761 | |
| full scorer, 20 ep (61k params) | 0.676 | 0.456 | 0.736 | changes 44/70 pairs; OOF-train score 0.84 → memorises clips |
| full scorer, 5 ep | 0.691 | 0.526 | 0.736 | +1 NEXAR clip, −short clips |
| scalars-only scorer | 0.696 | 0.456 | 0.761 | learns to keep the generator argmax |

*G. Native-frame refinement (±32 native frames, 3 dilated DW blocks, 54k params; trained on GT+random offsets)*

| variant | all | NEXAR | non-NEX |
|---|---|---|---|
| generator | 0.696 | 0.456 | 0.761 |
| refine top-1 / top-3 pairs, all clips | 0.651 | 0.479 | 0.698 |
| refine only clips >128 frames | 0.681 | 0.479 | 0.736 |

*Results (NEXAR official offline score)*

| recipe | CV 80 NEXAR clips (5-fold OOF, 2 seeds/fold) | all 349 (CV) | fixed 15 NEXAR val clips | fixed 70 |
|---|---|---|---|---|
| P2 4-seed ensemble (current submission recipe) | — | — | 0.500 | 0.757 |
| C0 control, 2-seed OOF ensemble | 0.561 | 0.704 | — | — |
| M_motion single model (motion as model input) | 0.576 ± 0.016 | 0.718 | 0.469 ± 0.015 (3 seeds) | 0.718 |
| C0 + X_ema + M_motion ensemble, plain decode | — | — | 0.510 | 0.738 |
| … + motion fusion + native snap (no prior) | **0.643** | **0.759** | **0.557** | 0.743 |
| … + position/gap prior | 0.643 | 0.756 | 0.557 | 0.743 |
| … + event-conditioned attribute head (no prior) | **0.655** | 0.760 | 0.535 | 0.733 |

**Source: [reports/stage2_phase_loss_report.md](/workspace/car-accident/reports/stage2_phase_loss_report.md)**


*Results (official offline score; fixed = mean ± std over 4 seeds; CV = pooled OOF, mean over 2 CV seeds)*

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

*Does the structured decoder fix long-video wrong-event selections? No: on balance it adds them.*

| model | catastrophic, direct | catastrophic, structured | fixed by structured | broken by structured |
|---|---|---|---|---|
| PH | 36 | 46 | 10 | 20 |
| PH+MS | 44 | 47 | 17 | 20 |

*Positional-artifact check (NEXAR collision ≈ 50 % of the clip)*

| model (decoder) | full | r = 0.3 | r = 0.5 | r = 0.7 |
|---|---|---|---|---|
| C0 (direct) | 0.551 | 0.571 | 0.553 | 0.580 |
| NT (direct) | 0.566 | 0.586 | 0.579 | 0.592 |
| PH (direct) | 0.557 | 0.578 | 0.566 | 0.556 |
| PH (structured, pre-declared) | 0.460 | 0.510 | 0.493 | 0.457 (coll. cat. 0.41) |

*Best use found: phase-loss model as an ensemble member with motion fusion*

| ensemble (+ motion fusion + snap) | CV NEXAR (80) | CV all (349) | fixed NEXAR (15) | fixed all (70) | fixed >1000 |
|---|---|---|---|---|---|
| current P2 4-seed ensemble, plain decode | — | — | 0.500 | 0.757 | 0.492 |
| C0 + X_ema + M_motion | 0.643 | 0.759 | 0.557 | 0.743 | 0.528 |
| C0 + X_ema + M_motion + NT (same-size control, no phase terms) | 0.641 | 0.751 | 0.589 | 0.753 | 0.562 |
| **C0 + X_ema + M_motion + PH** | **0.670** | **0.769** | **0.619** | **0.767** | **0.594** |

**Source: [reports/stage2_phase_supervision_study.md](/workspace/car-accident/reports/stage2_phase_supervision_study.md)**


*Results (5-fold CV; control A0 has 8 seeds; **bold rows have 8 seeds**, others 3)*

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

*Decoders (same checkpoints, 8 seeds; D0 = direct)*

| model | D0 direct | D1 transition | D2 local re-rank K4 / K8 (w from other folds) | D3 global structured |
|---|---|---|---|---|
| A2 CAT | **0.706** | 0.652 | 0.703 / 0.701 | 0.632 |
| R_ord1 | **0.705** | 0.636 | 0.698 / 0.697 | 0.626 |
| T_ord tr .75 | **0.706** | 0.679 | 0.702 / 0.702 | 0.654 |
| PH_repro | **0.690** | 0.681 | 0.690 / 0.690 | 0.666 |

*Gradient interaction (shared pyramid parameters, 8 fixed train batches, every epoch)*

| λ (ORD) | cos(direct, phase) init → ep1-3 → ep4-10 → ep11+ | abs(g_phase)/abs(g_direct), ep4-10 |
|---|---|---|
| 0.1 | −.04 → +.14 → +.27 → +.22 | 1.37 |
| 0.25 | −.04 → +.15 → +.26 → +.26 | 1.12 |
| 0.5 | −.04 → +.18 → +.33 → +.35 | 0.77 |
| 1 | −.04 → +.22 → +.39 → +.46 | 0.63 |
| 2 | −.04 → +.27 → +.46 → +.56 | 0.44 |

*v5 ensemble test (C0 + X_ema + M_motion unchanged + one 4th family; 8 members per fold; v5 motion fusion + snap)*

| 4th family | CV all (mean ± std, 4 draws) | NEXAR | >1000 | fixed all / NEXAR |
|---|---|---|---|---|
| NT control (A0) | 0.7568 ± 0.0019 | 0.638 | 0.631 | 0.746 / 0.557 |
| old PH recipe (PH_repro) | 0.7595 ± 0.0029 | 0.640 | 0.631 | 0.758 / 0.627 |
| CAT (A2) | 0.7600 ± 0.0060 | 0.637 | 0.630 | 0.744 / 0.534 |
| ORD (R_ord1) | 0.7594 ± 0.0024 | 0.645 | 0.636 | **0.779 / 0.604** |
| **ORD + transition .75** | **0.7636 ± 0.0029** | **0.650** | **0.642** | 0.753 / 0.589 |
| *v5's actual PH members (old seeds 0,1)* | *0.7688 (1 draw)* | *0.670* | *0.660* | *0.762 / 0.581* |
| *old NT seeds 0,1 (the v5 same-size control)* | *0.7510 (1 draw)* | *0.641* | *0.632* | *0.759 / 0.581* |

*Success criteria (8-seed finalists vs matched control A0)*

| criterion | A2 CAT | R_ord1 | T_ord tr .75 |
|---|---|---|---|
| ≥ +0.01 OOF | ✗ (+.007) | ✗ (+.005) | ✗ (+.007) |
| ≥ 4/5 folds | ✓ 5/5 | ✓ 4/5 | ✓ 4/5 |
| >1000 not −0.01 | ✓ +.004 | ✓ +.013 | ✓ −.006 |
| improves v5 when replacing PH members | recipe ≈ | recipe ≈ | recipe ✓ (+.004), actual members ✗ |

**Source: [reports/stage2_temporal_pyramid_framecount_experiments.md](/workspace/car-accident/reports/stage2_temporal_pyramid_framecount_experiments.md)**


*Overall validation*

| Run | Score | ENTRY | COLLISION | Side F1 | Evasion F1 | Entry normalized MAE | Collision normalized MAE | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.689 | 0.557 | 0.814 | 0.828 | 0.568 | 0.070 | 0.025 | 1,038,952 |
| P1 | 0.674 | 0.571 | 0.743 | 0.840 | 0.585 | 0.073 | 0.058 | 273,942 |
| P2 | 0.730 | 0.629 | 0.786 | 0.871 | 0.694 | 0.057 | 0.039 | 273,942 |
| P3 | 0.690 | 0.614 | 0.757 | 0.828 | 0.569 | 0.052 | 0.034 | 273,942 |
| P4 | 0.703 | 0.629 | 0.757 | 0.871 | 0.583 | 0.044 | 0.030 | 273,942 |

*Frame-count bins*

| Run | <150 | 150–300 | 300–500 | 500–750 | 750–1000 | >1000 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.780 (32) | 0.686 (21) | 0.450 (2) | 0.850 (1) | — (0) | 0.492 (14) | 0.689 |
| P1 | 0.749 (32) | 0.685 (21) | 0.725 (2) | 0.850 (1) | — (0) | 0.442 (14) | 0.674 |
| P2 | 0.802 (32) | 0.790 (21) | 0.725 (2) | 0.500 (1) | — (0) | 0.463 (14) | 0.730 |
| P3 | 0.759 (32) | 0.738 (21) | 0.625 (2) | 0.150 (1) | — (0) | 0.488 (14) | 0.690 |
| P4 | 0.786 (32) | 0.745 (21) | 0.475 (2) | 0.425 (1) | — (0) | 0.484 (14) | 0.703 |

*Frame-count bins*

| Run | Bin | n | ENTRY | COLLISION | Side F1 | Evasion F1 | Score | Entry norm MAE | Collision norm MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | <150 | 32 | 0.688 | 0.906 | 0.873 | 0.610 | 0.780 | 0.073 | 0.018 |
| B0 | 150–300 | 21 | 0.571 | 0.762 | 0.905 | 0.555 | 0.686 | 0.061 | 0.020 |
| B0 | 300–500 | 2 | 0.000 | 1.000 | 0.333 | 0.333 | 0.450 | 0.067 | 0.005 |
| B0 | 500–750 | 1 | 1.000 | 1.000 | 0.500 | 0.500 | 0.850 | 0.006 | 0.009 |
| B0 | 750–1000 | 0 | — | — | — | — | — | — | — |
| B0 | >1000 | 14 | 0.286 | 0.643 | 0.641 | 0.475 | 0.492 | 0.083 | 0.053 |
| P1 | <150 | 32 | 0.688 | 0.875 | 0.762 | 0.584 | 0.749 | 0.072 | 0.038 |
| P1 | 150–300 | 21 | 0.524 | 0.762 | 1.000 | 0.568 | 0.685 | 0.033 | 0.024 |
| P1 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 1.000 | 0.725 | 0.032 | 0.015 |
| P1 | 500–750 | 1 | 1.000 | 1.000 | 0.500 | 0.500 | 0.850 | 0.009 | 0.006 |
| P1 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P1 | >1000 | 14 | 0.357 | 0.357 | 0.785 | 0.497 | 0.442 | 0.143 | 0.161 |
| P2 | <150 | 32 | 0.656 | 0.969 | 0.875 | 0.681 | 0.802 | 0.059 | 0.019 |
| P2 | 150–300 | 21 | 0.714 | 0.810 | 0.952 | 0.762 | 0.790 | 0.024 | 0.024 |
| P2 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 1.000 | 0.725 | 0.033 | 0.007 |
| P2 | 500–750 | 1 | 1.000 | 0.000 | 0.500 | 0.500 | 0.500 | 0.002 | 0.020 |
| P2 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P2 | >1000 | 14 | 0.429 | 0.357 | 0.775 | 0.475 | 0.463 | 0.112 | 0.114 |
| P3 | <150 | 32 | 0.656 | 0.938 | 0.779 | 0.559 | 0.759 | 0.074 | 0.033 |
| P3 | 150–300 | 21 | 0.714 | 0.714 | 0.952 | 0.637 | 0.738 | 0.025 | 0.028 |
| P3 | 300–500 | 2 | 0.500 | 1.000 | 0.333 | 0.333 | 0.625 | 0.035 | 0.003 |
| P3 | 500–750 | 1 | 0.000 | 0.000 | 0.500 | 0.500 | 0.150 | 0.026 | 0.019 |
| P3 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P3 | >1000 | 14 | 0.429 | 0.429 | 0.775 | 0.475 | 0.488 | 0.044 | 0.051 |
| P4 | <150 | 32 | 0.719 | 0.938 | 0.779 | 0.593 | 0.786 | 0.049 | 0.021 |
| P4 | 150–300 | 21 | 0.714 | 0.714 | 0.952 | 0.679 | 0.745 | 0.024 | 0.037 |
| P4 | 300–500 | 2 | 0.000 | 0.500 | 1.000 | 1.000 | 0.475 | 0.040 | 0.020 |
| P4 | 500–750 | 1 | 1.000 | 0.000 | 0.500 | 0.000 | 0.425 | 0.009 | 0.019 |
| P4 | 750–1000 | 0 | — | — | — | — | — | — | — |
| P4 | >1000 | 14 | 0.357 | 0.500 | 0.928 | 0.300 | 0.484 | 0.067 | 0.043 |

*Sources and confounding*

| Run | AIHub (18) | CCD (17) | MM-AU (20) | Nexar (15) |
|---|---:|---:|---:|---:|
| B0 | 0.663 | 0.696 | 0.815 | 0.522 |
| P1 | 0.658 | 0.672 | 0.815 | 0.480 |
| P2 | 0.775 | 0.705 | 0.895 | 0.473 |
| P3 | 0.706 | 0.673 | 0.829 | 0.473 |
| P4 | 0.709 | 0.686 | 0.859 | 0.486 |

*Sources and confounding*

| Run | ENTRY hits | COLLISION hits | Entry norm MAE | Collision norm MAE |
|---|---:|---:|---:|---:|
| B0 | 5/15 | 10/15 | 0.078 | 0.051 |
| P1 | 6/15 | 6/15 | 0.134 | 0.151 |
| P2 | 7/15 | 5/15 | 0.105 | 0.108 |
| P3 | 6/15 | 6/15 | 0.043 | 0.049 |
| P4 | 6/15 | 7/15 | 0.063 | 0.041 |

*Sources and confounding*

| Run | Global rho | Nexar rho | Global entry-error rho | Global collision-error rho |
|---|---:|---:|---:|---:|
| B0 | -0.320 | -0.109 | -0.113 | 0.080 |
| P1 | -0.277 | -0.112 | -0.113 | 0.177 |
| P2 | -0.250 | -0.360 | -0.187 | 0.066 |
| P3 | -0.239 | -0.258 | -0.268 | -0.083 |
| P4 | -0.284 | -0.518 | -0.302 | 0.009 |

*Overfitting and resource cost*

| Run | Best epoch | Last epoch | Gap (epochs) | Train selection score at best weights | Val selection score | Head ms/video | Peak allocated VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 14 | 30 | 16 | — | 0.464 | 0.157 | 616.4 |
| P1 | 11 | 18 | 7 | 0.870 | 0.444 | 0.660 | 207.7 |
| P2 | 7 | 14 | 7 | 0.716 | 0.475 | 0.636 | 489.4 |
| P3 | 6 | 13 | 7 | 0.607 | 0.455 | 0.642 | 484.3 |
| P4 | 5 | 12 | 7 | 0.654 | 0.453 | 0.614 | 830.1 |

*Sampling-density robustness*

| Run | Score range | Entry normalized prediction SD | Collision normalized prediction SD | Entry mean frame range | Collision mean frame range |
|---|---:|---:|---:|---:|---:|
| B0 | 0.028 | 0.0184 | 0.0082 | 17.6 | 9.3 |
| P1 | 0.055 | 0.0182 | 0.0185 | 24.1 | 17.4 |
| P2 | 0.028 | 0.0131 | 0.0102 | 27.8 | 19.4 |
| P3 | 0.024 | 0.0077 | 0.0087 | 8.5 | 6.7 |
| P4 | 0.033 | 0.0094 | 0.0036 | 20.7 | 4.7 |

**Source: [stage2/reports/stage2_dinov3_experiments_report.md](/workspace/car-accident/stage2/reports/stage2_dinov3_experiments_report.md)**


*Reproduction*

| Baseline | ENTRY | COLL | side F1 | evasion F1 | Score |
|---|---:|---:|---:|---:|---:|
| Recorded `training_report.json` | 0.40 | 0.70 | 0.8199 | 0.6800 | 0.6100 |
| Reproduction, seed 0 | 0.46 | 0.70 | 0.8599 | 0.5495 | 0.6174 |
| Reproduction, seed 42 | 0.40 | 0.64 | 0.8182 | 0.6566 | 0.5852 |

*Changes implemented*

| Component | Prompt | File |
|---|---|---|
| Multi-rate temporal differences `D_s[t]=F[t]-F[t-s]`, gated fusion, boundary-clamped | §6 | `modules.py:MultiRateDifference` |
| Event-state auxiliary heads, BEFORE/BOUNDARY/AFTER and PRE/CONTACT/POST, radius in sampled positions | §9 | `objective.py:state_aux_loss` |
| Two-view temporal consistency, Jensen-Shannon over probability mass binned into shared original-frame coordinates | §12 | `objective.py:consistency_loss` |
| Local temporal warping, frame duplication, random drop, random cadence | §11 | `sampling.py` |
| P25/P50/P75/P90 error distributions | §17 | `metrics.py:error_distribution` |
| Sampling-density instability report | §18 | `robustness.py` |
| Directional and conditional failure breakdown | §21 | `failure_analysis.py` |

*Experiment table*

| Experiment | ENTRY | COLL | side F1 | evasion F1 | Score | Params |
|---|---:|---:|---:|---:|---:|---:|
| E2c local attention | 0.48 | 0.82 | 0.779 | 0.643 | **0.6683** | 1.04M |
| E8 event-conditioned | 0.42 | 0.80 | 0.820 | 0.619 | 0.6428 | 1.62M |
| E7 temporal consistency | 0.42 | 0.74 | 0.860 | 0.694 | 0.6391 | 1.41M |
| E4 soft targets | 0.38 | 0.76 | 0.880 | 0.635 | 0.6263 | 1.41M |
| E6 sampling augmentation | 0.42 | 0.74 | 0.839 | 0.576 | 0.6182 | 1.41M |
| *Baseline, seed 0* | *0.46* | *0.70* | *0.860* | *0.550* | *0.6174* | *0.81M* |
| E2b TemporalMaxer | 0.40 | 0.66 | 0.860 | 0.594 | 0.5891 | 0.67M |
| E5 state aux | 0.40 | 0.70 | 0.778 | 0.579 | 0.5884 | 1.41M |
| E3 multi-rate diff | 0.38 | 0.64 | 0.900 | 0.612 | 0.5839 | 1.41M |
| *Baseline, seed 42* | *0.40* | *0.64* | *0.818* | *0.657* | *0.5852* | *0.81M* |
| E1 dense event head | 0.28 | 0.62 | 0.860 | 0.740 | 0.5550 | 0.44M |
| E2a dilated conv | 0.24 | 0.62 | 0.880 | 0.740 | 0.5440 | 0.81M |
| B0 pooled baseline | 0.22 | 0.14 | 0.797 | 0.700 | 0.3505 | 0.48M |

*Experiment table*

| Run | ENTRY p50 | ENTRY p90 | COLL p50 | COLL p90 |
|---|---:|---:|---:|---:|
| E2c | 5.0 fr / 0.0372 | 59.3 fr / 0.2339 | 1.0 fr / 0.0067 | 8.0 fr / 0.0403 |
| E7 | 6.0 fr / 0.0336 | 40.2 fr / 0.2265 | 2.0 fr / 0.0134 | 19.7 fr / 0.1083 |
| E8 | 7.0 fr / 0.0408 | 77.9 fr / 0.2449 | 2.0 fr / 0.0124 | 18.0 fr / 0.0757 |

*FPS-invariance robustness*

| Run | std128 | dense192 | 96 | drop20 | drop40 | warp | **mean** | **worst** | range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E7 consistency | 0.6391 | 0.6255 | — | 0.6054 | 0.5365 | 0.5935 | **0.6088** | **0.5365** | 0.1166 |
| E8 event-cond | 0.6428 | 0.5770 | — | 0.6008 | 0.5429 | 0.5419 | 0.5906 | 0.5419 | **0.1009** |
| E2c attention | **0.6683** | 0.6163 | 0.6099 | 0.5859 | 0.4861 | 0.5315 | 0.5830 | 0.4861 | 0.1822 |
| E6 augmentation | 0.6182 | 0.5972 | — | 0.5617 | 0.5314 | 0.5002 | 0.5648 | 0.5002 | 0.1180 |
| E3 multi-rate | 0.5839 | 0.5574 | — | 0.5499 | 0.4890 | 0.5254 | 0.5471 | 0.4890 | 0.0949 |

*Failure analysis*

| Condition | group | rest |
|---|---:|---:|
| tight ENTRY→COLLISION gap (<=10 frames, n=11) | 0.1105 | 0.0752 |
| short clip (<200 frames, n=34) | 0.0904 | 0.0671 |
| early event (position < 0.4, n=6) | 0.0874 | 0.0823 |

*Runtime*

| Run | wall | head | peak VRAM | params |
|---|---:|---:|---:|---:|
| E2c | 119 s | 0.20 ms/video | 567 MiB | 1.04M |
| E7 | 161 s | 0.21 ms/video | 929 MiB | 1.41M |
| E8 | 166 s | 0.26 ms/video | 932 MiB | 1.62M |

**Source: [stage2/reports/stage2_geometry_dino_spotting_experiments.md](/workspace/car-accident/stage2/reports/stage2_geometry_dino_spotting_experiments.md)**


*Fixed-split results*

| ID | Head / target | ENTRY | COLLISION | Side F1 | Evasion F1 | Score | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 | local attention / normalized soft | 0.44 | 0.74 | 0.860 | 0.616 | **0.634** | 1.04M |
| E0-fpsblind | dilated conv / normalized soft | **0.50** | 0.66 | 0.780 | **0.657** | 0.621 | 0.81M |
| E1 displacement | encoder-decoder / radius+offset | **0.52** | 0.60 | 0.859 | 0.600 | 0.611 | 1.18M |
| exact-anchor | baseline / exact CE | 0.42 | 0.68 | 0.880 | 0.618 | 0.610 | 0.81M |
| normalized 1% | baseline / normalized soft | 0.44 | 0.64 | 0.819 | 0.720 | 0.609 | 0.81M |
| E2 | TemporalMaxer / normalized soft | 0.34 | **0.76** | **0.880** | 0.594 | 0.606 | **0.56M** |
| E1 classification | encoder-decoder / normalized soft | 0.40 | 0.66 | 0.788 | 0.559 | 0.573 | 1.18M |
| E4 | event-conditioned TemporalMaxer | 0.38 | 0.74 | 0.653 | 0.548 | 0.572 | 0.76M |
| soft sampled-index | baseline / index soft | 0.40 | 0.60 | 0.779 | 0.678 | 0.569 | 0.81M |
| E2 displacement | TemporalMaxer / radius+offset | 0.26 | 0.68 | 0.778 | 0.720 | 0.554 | 0.56M |
| E3 displacement | local attention / radius+offset | 0.34 | 0.56 | 0.840 | 0.699 | 0.546 | 1.04M |

*E5 and robustness*

| Source | N | ENTRY | COLLISION |
|---|---:|---:|---:|
| AIHUB | 18 | 0.0350 | 0.0336 |
| CCD | 16 | 0.0855 | 0.0204 |
| NEXAR | 16 | 0.0685 | 0.0514 |

**Source: [stage2/long_video_experiments/REPORT.md](/workspace/car-accident/stage2/long_video_experiments/REPORT.md)**


*Scope*

| Model and view | Selected epoch | FPS-blind selection | Offline score | ENTRY | COLLISION | Side F1 | Evasion F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing soft target, 128 | 14 | 0.4643 | **0.6893** | 0.557 | **0.814** | 0.828 | 0.568 |
| Same checkpoint, 192 | 14 | 0.4199 | 0.6749 | 0.571 | 0.743 | 0.857 | 0.576 |
| Tiled local 32 / step 16, 192 | 15 | 0.4194 | 0.6694 | 0.557 | 0.743 | 0.753 | 0.676 |
| Tiled local 64 / step 32, 192 | 9 | 0.4083 | 0.6483 | 0.529 | 0.729 | 0.726 | 0.663 |
| Exact ENTRY and COLLISION, 128 | 6 | **0.4749** | 0.6849 | **0.571** | 0.743 | **0.886** | 0.614 |
| Exact ENTRY, soft COLLISION, 128 | 28 | 0.4683 | 0.6733 | 0.543 | 0.729 | 0.843 | **0.679** |

*NEXAR validation (15 clips)*

| Model | ENTRY hits | COLLISION hits | Median ENTRY error | Median COLLISION error |
|---|---:|---:|---:|---:|
| Existing soft, 128 | 5 | **10** | 41 frames | **5 frames** |
| Existing soft, 192 | 6 | 6 | 21 frames | 11 frames |
| Tiled 32, 192 | 5 | 7 | 18 frames | 11 frames |
| Tiled 64, 192 | 5 | 6 | 17 frames | 13 frames |
| Exact both, 128 | **7** | 8 | 13 frames | 6 frames |
| Exact ENTRY / soft COLLISION, 128 | **7** | 6 | **12 frames** | 17 frames |

**Source: [stage2/nexar65_experiments/REPORT.md](/workspace/car-accident/stage2/nexar65_experiments/REPORT.md)**


*All training runs*

| Run | Seed | Train n | Selected epoch | NEXAR score | ENTRY hits /15 | COLLISION hits /15 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| LN_exact | 0 | 279 | 8 | 0.5509 | 5 | 8 | 0.6796 |
| LN_exact | 1 | 279 | 6 | 0.4621 | 4 | 6 | 0.6556 |
| LN_exact | 2 | 279 | 17 | 0.5820 | 7 | 9 | 0.6505 |
| LN_frame6 | 0 | 279 | 18 | 0.4889 | 5 | 7 | 0.7216 |
| LN_nexar_ft | 0 | 65 | 3 | 0.5890 | 8 | 8 | 0.6967 |
| LN_nexar_prior_ft | 0 | 65 | 10 | 0.6415 | 9 | 10 | 0.6921 |
| LN_prior_repeat2 | 0 | 279 | 11 | 0.6600 | 8 | 12 | 0.6581 |
| LN_repeat2 | 0 | 279 | 29 | 0.5779 | 8 | 8 | 0.7091 |
| LN_soft | 0 | 279 | 5 | 0.5361 | 6 | 9 | 0.6879 |
| LN_soft | 1 | 279 | 9 | 0.5161 | 8 | 7 | 0.6955 |
| LN_soft | 2 | 279 | 10 | 0.5511 | 6 | 8 | 0.6683 |
| TCN_frame6 | 0 | 279 | 30 | 0.5616 | 6 | 8 | 0.7343 |
| TCN_frame6 | 1 | 279 | 5 | 0.5242 | 6 | 6 | 0.6161 |
| TCN_frame6 | 2 | 279 | 17 | 0.5674 | 7 | 8 | 0.7088 |

*Reloaded ensembles*

| Ensemble | NEXAR | ENTRY /15 | COLLISION /15 | Side F1 | Evasion F1 | Overall | non-NEXAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_three_no_prior | 0.5805 | 7 | 8 | 0.9333 | 0.6032 | 0.7451 | 0.7893 |
| first_three_prior | 0.7438 | 11 | 11 | 0.9333 | 0.6032 | 0.6701 | 0.6493 |
| first_three_prior_seed1 | 0.6295 | 11 | 8 | 0.8661 | 0.3750 | 0.6315 | 0.6274 |
| first_three_prior_seed2 | 0.6461 | 9 | 9 | 0.8661 | 0.6411 | 0.6480 | 0.6486 |
| nine_heads_prior | 0.7104 | 11 | 10 | 0.8661 | 0.6032 | 0.6437 | 0.6253 |

**Source: [stage2/length_gated_experiments/REPORT.md](/workspace/car-accident/stage2/length_gated_experiments/REPORT.md)**


*Source results*

| Slice | n | Original P2 | Hybrid: seed 0 | Hybrid: all seeds |
|---|---:|---:|---:|---:|
| **All validation** | **70** | 0.7567 | **0.8089** | **0.8017** |
| NEXAR | 15 | 0.5004 | **0.7438** | **0.7104** |
| AIHub | 18 | **0.7738** | 0.7738 | 0.7738 |
| CCD | 17 | **0.7950** | 0.7950 | 0.7950 |
| MM-AU | 20 | **0.8902** | 0.8902 | 0.8902 |
| All non-NEXAR | 55 | **0.8257** | 0.8257 | 0.8257 |
