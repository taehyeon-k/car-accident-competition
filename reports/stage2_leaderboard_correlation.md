# Stage 2: validation vs leaderboard, and what the evaluation set looks like (2026-09-27)

## Leaderboard history (Stage 2)
| submission | LB S2 | offline CV all / NEXAR (native) | note |
|---|---|---|---|
| v1 E3 ASFormer | 0.4410 | fixed 0.689 | |
| v3 P2 refit | 0.4618 | fixed 0.757 (proxy) | |
| NEXAR specialist | 0.437 | fixed 0.670 / NEXAR 0.744 | tuned on 15 NEXAR val clips |
| length-gated hybrid | 0.4154 | fixed 0.802 / NEXAR 0.710 | changes only clips > 500 native frames |
| v5 LC-v2 15-head + motion | 0.5314 | CV 0.769 / 0.670 | first CV-selected model |
| v7 E4+E2+XN4 (full refit) | 0.5464 | CV 0.784 / 0.717 | |
| **v8 = v7 + temporal-rate (stride) augmentation** | **0.5929** | CV 0.787 / 0.707 | |

## Which validation metric predicts the leaderboard? (v7 -> v8, same code, same members, only stride aug differs)
LB Δ = **+0.047**.
| metric | v7 | v8 | Δ | matches? |
|---|---|---|---|---|
| native CV (goal_eval) | 0.784 | 0.787 | +0.003 | no |
| native NEXAR | 0.717 | 0.707 | −0.010 | wrong sign |
| LOSO ensemble mean (E4-type, 3 seeds) | 0.703 | 0.711 | +0.008 | no |
| plain CV at 1/2 frame rate | 0.720 | 0.759 | +0.039 | close |
| plain CV at 1/3 frame rate | 0.648 | 0.718 | +0.070 | too large |
| **~10 fps view** (CCD native 10 fps, MMAU/NEXAR 30→10 fps, AIHUB 15 fps = mean of native and 7.5 fps) | 0.730 | 0.777 | **+0.047** | **yes** |
Offset: LB ≈ 10-fps view − 0.184 for both (0.730 → 0.546, 0.777 → 0.593). Two points only: this is a consistent explanation, not
a fitted law; the next submission tests it.

## Traits of the evaluation set this implies
1. **Frame rate ≈ 10 fps.** Evidence: the v7 → v8 gain is explained exactly by the 10 fps view and not by any native metric; every
   DACON sample video (Stage 1 and Stage 2) is 1280×720 at 10 fps, 50 frames (Stage 2 sample: collision at frames 30–41, CCD-like);
   Stage 3 is stated to be 10 Hz. Our training sources are 15 fps (AIHUB) and 30 fps (MMAU, NEXAR); only CCD is 10 fps.
2. ~~Long clips exist.~~ **Weak.** The hybrid's short-clip branch was a 279-clip P2 proxy while the scored v3 was a 349-clip refit, so
   "only long clips changed" is not a clean comparison (correction from the independent review).
3. **A large, roughly constant domain/label shift (≈ 0.18)** on top of the frame rate: larger than any held-out source in LOSO
   (ensemble ≈ 0.08 below CV). The test is unlike every training source, as the organisers state.
4. Consequence already seen: models tuned on the 15 NEXAR val clips lost (specialist, hybrid); CV-selected models won (v5); the
   frame-rate-robust model won the most (v8).

## Re-ranking current candidates by the 10 fps view (plain decoding, 3 seeds per family)
| ensemble | native | 1/3 fps | 10 fps view | predicted LB (view − 0.184) | NEXAR @10 fps |
|---|---|---|---|---|---|
| v7 recipe | 0.772 | 0.648 | 0.730 | 0.546 (actual 0.5464) | 0.600 |
| v8 recipe | 0.775 | 0.718 | **0.777** | 0.593 (actual 0.5929) | **0.685** |
| v9 recipe (+EMA) | 0.771 | 0.696 | 0.761 | 0.577 | 0.652 |
| E4_sa + E2_sa_xu + XN4_sa_xu | **0.791** | 0.729 | 0.775 | 0.591 | 0.647 |
| XU_mc + E2_sa_xu + XN4_sa_xu | 0.779 | 0.732 | 0.772 | 0.588 | 0.664 |
The metadata-expansion ensembles (X1) win at native frame rate but **not** in the 10 fps view: the extra clips are only trained at
their native rate (stride augmentation is applied to the labelled clips only), which dilutes the frame-rate robustness at 10 fps.

## Implications
* Select Stage 2 models by the **10 fps view** (and 1/3-rate CV), not native CV. Native CV and NEXAR-native slices have not predicted
  the last step.
* Train toward 10 fps: stride-augment the extra clips too, and consider sampling each clip at its ~10 fps rate more often
  (30 fps sources: stride 3; AIHUB: stride 1–2; CCD: native).
* Everything must stay FPS-blind at inference (frame counts only): the model must be robust to the test frame rate, not told it.

## Update: v5 and v6 added (4 leaderboard points; `stage2/generalization/v56_view.py`, CV proxies seeds 0–1, plain decoding)
| recipe | LB | native CV | CV at 1/3 frame rate | ~10 fps view |
|---|---|---|---|---|
| v6 (event-specific 18 heads) | 0.5277 | 0.771 | 0.634 | 0.726 |
| v5 (LC-v2 15 heads) | 0.5314 | 0.742 | 0.636 | 0.703 |
| v7 | 0.5464 | 0.772 | 0.648 | 0.730 |
| v8 | 0.5929 | 0.775 | 0.718 | 0.777 |
| Pearson vs LB | | 0.51 | **0.99** | 0.94 |
| rank order correct? | | no (v5 < v6) | **yes** | no (v5/v6 swapped, LB gap 0.004) |
**Plain CV at 1/3 of the native frame rate ranks all four submissions correctly** (native CV does not). Caveats: four related
submissions, and 1/3-rate was picked after looking at several views (native, 1/2, 1/3, 10 fps view), so r = 0.99 is optimistic.
Working selection rule from now on: **1/3-rate CV first, 10 fps view second, native CV third**, on duplicate-clean clips.
Interpretation: the hidden clips behave like our clips at ~1/3 of their native rate (e.g. ~10 fps footage vs our 30 fps sources);
the exact cadence is still not observable.

## Update 2026-09-27 evening: v10 = 0.5590 — the ⅓-rate rule failed out of sample
v10 (v8 families + metadata-labelled MM-AU/CCD extras) scored **0.5590**, −0.034 vs v8, although duplicate-clean CV said +0.004 native,
+0.019 at ⅓ rate (forecast 0.607) and ≈ −0.002 in the 10 fps view. Per source (native): AIHUB −0.015, CCD −0.008, MMAU −0.009,
NEXAR +0.035, ENTRY −0.03…−0.05 on the three non-NEXAR sources; the ⅓-rate gain came from AIHUB and CCD at 3.3 fps (unrealistic:
the DACON CCD samples are native 10 fps). No offline metric predicted the drop's size. Prime suspect: the pseudo-labelled extras
(teacher ENTRY / metadata COLLISION conventions, or same-source similarity inflating CV). **v11 / v12 contain the same extras, so
their forecasts are withdrawn.** Five LB points now: v5 0.5314, v6 0.5277, v7 0.5464, v8 0.5929, v10 0.5590.
* **v12 = 0.5635** (XSbU + ViT-B family; contains the MM-AU/CCD extras). Offline it was the best at every frame rate (⅓ rate 0.745 vs
  v8 0.710). Together with v10 (0.5590) this is the second submission with extras below v8 (0.5929): the extras-trained recipes lose
  ~0.03 on the LB despite better CV. Six LB points: v5 .5314, v6 .5277, v7 .5464, v8 .5929, v10 .5590, v12 .5635.
