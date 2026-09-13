# Stage2 architecture modification

> 2026-09-13 implementation update: V-JEPA sampling is **index-only**, per the
> user correction: 16 frames at native stride 4, clip-start stride 48, deterministic
> tail coverage, and the existing rounded-linspace policy for clips shorter than
> 61 frames. No FPS is required at inference. See [JOINT.md](JOINT.md).

## 1. Object tracking criteria.

- Replace existing object criteria
- New criteria
    1. Detection Confidence
        
        ```c
        conf = float(detection.score)
        if conf < 0.20:
            continue
        conf_score = (conf - 0.20) / 0.80
        conf_score = np.clip(conf_score, 0.0, 1.0)
        ```
        
    2. Ego-lane proximity
        
        ```c
        x1, y1, x2, y2 = box
        
        bottom_center_x = (x1 + x2) / 2
        bottom_y = y2
        
        x = bottom_center_x / frame_width
        y = bottom_y / frame_height
        
        def ego_lane_score(x, y):
            y = np.clip(y, 0.0, 1.0)
        
            # approximate road horizon
            horizon_y = 0.40
        
            if y < horizon_y:
                return 0.0
        
            t = (y - horizon_y) / (1.0 - horizon_y)
        
            half_width_top = 0.06
            half_width_bottom = 0.30
        
            half_width = (
                half_width_top
                + t * (half_width_bottom - half_width_top)
            )
        
            center = 0.5
            lateral_distance = abs(x - center)
        
            score = 1.0 - lateral_distance / half_width
        
            return float(np.clip(score, 0.0, 1.0))
        ```
        
    3. X-axis object movement
        - Use **lateral x-axis motion**. Because test-time FPS may be unavailable, measure motion in normalized image coordinates **per frame**, then convert it to a within-video relative rank rather than an absolute physical-speed threshold.
        - Use absolute x-motion magnitude because both left-to-right and right-to-left entering vehicles can be relevant.
        
        ```c
        def estimate_x_motion_per_frame(history, frame_width):
            if len(history) < 2:
                return None
        
            history = history[-5:]
            frames = np.array([h.frame for h in history], dtype=np.float32)
            xs = np.array([h.center_x / frame_width for h in history], dtype=np.float32)
        
            frame_offsets = frames - frames[0]
            if frame_offsets[-1] < 1e-6:
                return None
        
            vx_per_frame = np.polyfit(frame_offsets, xs, 1)[0]
            return float(abs(vx_per_frame))
        
        x_motion_score = percentile_rank(
            x_motion_per_frame,
            valid_x_motion_values_in_video,
        )
        ```
        
        - The absolute per-frame magnitude may change with FPS, but within one constant-FPS video FPS is a common scale factor, so the relative ordering of vehicles is preserved.
        - Do not use a fixed threshold such as `vx / 0.5`; that would depend on FPS and source characteristics.
    4. Bounding Box growth
        - Apply the same FPS-independent principle: compute normalized log-area growth **per frame**, then convert positive growth to a within-video relative rank.
        
        ```c
        area = (
            (x2 - x1) / width
            * (y2 - y1) / height
        )
        
        growth_per_frame = (
            np.log(curr_area + 1e-8)
            - np.log(prev_area + 1e-8)
        ) / max(curr_frame - prev_frame, 1)
        
        positive_growth = max(float(growth_per_frame), 0.0)
        growth_score = percentile_rank(
            positive_growth,
            valid_positive_growth_values_in_video,
        )
        ```
        
- Total score
    
    ```c
    terms = [
        (0.20, confidence_score, True),
        (0.40, lane_score, True),
        (0.30, x_motion_score, has_motion),
        (0.10, growth_score, has_growth),
    ]
    
    numerator = sum(w * value for w, value, valid in terms if valid)
    denominator = sum(w for w, _, valid in terms if valid)
    
    priority = numerator / max(denominator, 1e-6)
    ```
    

## 2. Hybrid temporal refinement after V-JEPA fusion

- **Goal:** replace the current four sequential dilated Conv1D blocks (`dilation = 1, 2, 4, 8`) with a stronger temporal refinement module that combines adaptive local self-attention and multi-scale convolution.
- **Why modify the current temporal head**
    - V-JEPA 2.1 already provides sparse/global temporal context through temporal cross-attention, so another deep global Transformer would be partly redundant.
    - Pure dilated convolution is efficient and has a good locality bias for precise ENTRY/COLLISION spotting, but uses fixed learned filters and cannot adaptively choose which neighboring frames matter based on content.
    - The preferred design therefore keeps convolutional locality while adding shallow local self-attention for content-dependent temporal interaction.
- **Recommended architecture**
    
    ```
    Fused local/global sequence H [B, T, 384]
            ↓
    Hybrid Temporal Block ×2
       ├─ Local MHSA: ±16-frame window
       │    d_model = 384
       │    heads = 6
       │    head_dim = 64
       │
       └─ Multi-scale temporal Conv branch
            depthwise Conv1D k=5, dilation=1
            depthwise Conv1D k=5, dilation=2
            depthwise Conv1D k=5, dilation=4
                 ↓
          branch fusion / projection
                 ↓
              residual
                 ↓
          FFN 384 → 1536 → 384
                 ↓
              residual
            ↓
    Small temporal Conv refinement (k=5, dilation=1 or 2)
            ↓
    ENTRY / COLLISION frame heads
    ```
    
- **Local self-attention branch**
    - Each frame attends only to a local temporal neighborhood instead of the complete video.
    - Initial recommendation: radius `16`, giving a `33-frame` attention window.
    - Unlike Conv1D, attention can dynamically emphasize specific neighboring states according to feature similarity and event context.
- **Multi-scale temporal convolution branch**
    - Run multiple temporal scales in parallel rather than only stacking increasingly dilated blocks sequentially.
    - Recommended parallel branches: `kernel=5` with `dilation={1,2,4}`.
    - `d=1` captures very local transitions and collision boundaries.
    - `d=2` captures short motion evolution.
    - `d=4` captures broader approach/entry trends.
    - Fuse the parallel outputs with concatenation + projection or a learned weighted sum back to `384D`.

## 3. Event-conditioned Entry-Side and Evasion-Space heads

- **Decision:** keep the current ENTRY and COLLISION frame-localization heads unchanged. Modify only the attribute heads so they receive event-specific spatial/object evidence instead of relying mostly on compressed temporal embeddings and global averages.
- **Core principle**
    - ENTRY/COLLISION heads answer **when** the event occurs.
    - Entry-side and evasion-space heads should then inspect **what and where** around the corresponding event.
    - Do not simply deepen the final MLP. Preserve and expose spatial/object information that is currently discarded before attribute classification.
    - **Gradient-flow decision:** attribute losses must **not** move the predicted ENTRY/COLLISION distributions. Use stop-gradient event probabilities when forming event-centered embeddings/tokens:
        
        $$
        P_E^{attr}(t)=\operatorname{stopgrad}(P_E(t)), \qquad P_C^{attr}(t)=\operatorname{stopgrad}(P_C(t))
        $$
        
    - Gradients from `entry_side` and `evasion_space` still flow through the selected scene/object features, attribute cross-attention, and attribute MLPs; only the path back through `P_E` / `P_C` is detached. This prevents an attribute objective from improving itself by shifting the event-localization distribution away from the true event.
- **Current bottleneck**
    - Before temporal modeling, each frame has `7` DINO scene tokens and up to `12` object tokens after the spatial Transformer.
    - The current model keeps only the updated CLS token (`[:, 0]`) for the main temporal path, so the attribute heads must infer detailed spatial relationships from a single compressed `384D` representation.
    - This is especially limiting for `entry_side`, which depends on the entering object's lateral position/motion, and `evasion_space`, which depends on the spatial arrangement of free road, obstacles, and nearby vehicles.
- **3-A. Entry-side head modification**
    - Keep the existing soft ENTRY distribution:
        
        $$
        P_E(t)=\mathrm{softmax}(L_E(t))
        $$
        
    - Keep the existing ENTRY event embedding:
        
        $$
        h_E=\sum_t P_E(t)h_t
        $$
        
    - In addition, preserve the post-spatial-Transformer scene/object tokens for each frame instead of discarding them completely.
    - Build ENTRY-centered spatial/object tokens by temporally pooling them with the predicted ENTRY distribution. For object tokens:
        
        $$
        O_E=\sum_t P_E(t)O_t, \qquad O_E\in\mathbb{R}^{12\times384}
        $$
        
    - Optionally do the same for the six spatial scene cells, producing `6 × 384` ENTRY-centered scene tokens.
    - Recommended input to the attribute attention block: `6 scene + 12 object = 18` ENTRY-local spatial tokens.
    - Use the ENTRY embedding as a query over these tokens:
        
        $$
        z_E=\mathrm{CrossAttention}(q=h_E,\;K=X_E,\;V=X_E)
        $$
        
    - Here, `X_E` contains the ENTRY-centered scene and object tokens. Attention allows the head to assign high weight to the vehicle/scene region most relevant to the entry event instead of averaging all visible objects equally.
    - The object tokens already combine DINO ROI appearance and geometry, so the head can use cues such as lateral x-position, motion, proximity, bounding-box evolution, and appearance to determine whether the relevant vehicle entered from the LEFT or RIGHT.
    - Recommended lightweight implementation:
        
        ```
        ENTRY logits → soft ENTRY distribution
                         ↓
        Temporal hidden states → ENTRY embedding h_E [384]
                         │
                         │ query
                         ▼
        ENTRY-local spatial/object tokens [18, 384]
               (6 scene + 12 objects)
                         ↓
              1-layer Cross-Attention
              d_model=384, heads=6
                         ↓
              ENTRY spatial embedding
                         ↓
                residual / projection
                         ↓
                   MLP 384→128→2
                         ↓
                     LEFT / RIGHT
        ```
        
    - Prefer this over simply concatenating a whole-video V-JEPA mean, because the relevant evidence for side classification is strongly localized around ENTRY and is primarily spatial/object-centric.
- **3-B. Evasion-space head modification**
    - Keep the existing soft COLLISION distribution:
        
        $$
        P_C(t)=\mathrm{softmax}(L_C(t))
        $$
        
    - Keep the current COLLISION event embedding:
        
        $$
        h_C=\sum_t P_C(t)h_t
        $$
        
    - Preserve the six DINO spatial cells separately around COLLISION instead of averaging them into one scene vector.
    - Current averaging loses spatial arrangement. For example, `left free / right blocked` and `left blocked / right free` can become too similar after mean pooling.
    - Construct COLLISION-centered scene tokens:
        
        $$
        S_C=\sum_t P_C(t)S_t, \qquad S_C\in\mathbb{R}^{6\times384}
        $$
        
    - Also construct COLLISION-centered object tokens:
        
        $$
        O_C=\sum_t P_C(t)O_t, \qquad O_C\in\mathbb{R}^{12\times384}
        $$
        
    - Concatenate them into `18` collision-local spatial tokens and use the collision embedding as the query:
        
        $$
        z_C=\mathrm{CrossAttention}(q=h_C,\;K=[S_C;O_C],\;V=[S_C;O_C])
        $$
        
    - This lets the evasion head dynamically inspect the spatial regions and objects relevant to whether usable evasion space existed: free roadway, nearby vehicles, road edges, lane regions, barriers, or other obstacles.
    - Recommended lightweight implementation:
        
        ```
        COLLISION logits → soft COLLISION distribution
                            ↓
        Temporal hidden states → COLLISION embedding h_C [384]
                            │
                            │ query
                            ▼
        COLLISION-local spatial/object tokens [18, 384]
                  (6 scene + 12 objects)
                            ↓
                 1-layer Cross-Attention
                 d_model=384, heads=6
                            ↓
               collision spatial embedding
                            ↓
                   residual / projection
                            ↓
                      MLP 384→128→1
                            ↓
                     EVASION 0 / 1
        ```
        
- **Why this is preferable to deeper MLP heads**
    - A deeper MLP only increases capacity after information has already been compressed.
    - It cannot reconstruct object identity or left/right/free-space layout if those relationships were lost during earlier pooling.
    - Event-conditioned spatial attention changes the **information available to the classifier**, not merely the number of classifier parameters.

## 4. Global top-12 object selection with persistent track identity

- **Problem:** the current cache selects the top 12 objects independently at every frame. Because ranking changes over time, slot `i` does not necessarily refer to the same physical vehicle across frames. This destroys persistent track identity and makes temporal object reasoning unreliable.
- **Decision:** select the 12 most relevant **tracks globally for the whole video/crop**, then assign each selected track to a fixed slot for its entire lifetime.
- **Global selection procedure**
    1. Run the Hungarian tracker over the complete video first.
    2. Compute the per-observation relevance score using the revised criteria from toggle #1.
    3. Aggregate observation-level scores into a single track-level score. Recommended starting point: use a robust high-percentile or top-k mean rather than a simple full-track mean, so an accident-relevant vehicle that becomes important only near ENTRY/COLLISION is not diluted by earlier low-score frames.
    4. Rank all tracks by this global track score and retain the top `12` tracks.
    5. Assign each retained track a persistent slot `0..11`.
    6. At every frame, write that track's ROI/geometry into the same slot when visible; otherwise leave the slot invalid and zero-filled.
- **Resulting tensor semantics**
    
    ```
    slot 0 = track A throughout the video
    slot 1 = track B throughout the video
    ...
    slot 11 = track L throughout the video
    
    frame t:     [A, B, -, D, ...]
    frame t+1:   [A, B, C, D, ...]
    frame t+2:   [A, -, C, D, ...]
    
    '-' means that tracked object is not visible at that frame,
    not that the slot is reassigned to another object.
    ```
    
- **Why this matters**
    - Preserves physical object identity across time.
    - Makes geometry trajectories such as x/y movement, area growth, depth evolution, and continuity semantically coherent.
    - Allows the event-conditioned entry-side/evasion heads in toggle #3 to pool or attend to object trajectories without mixing unrelated vehicles in the same slot.
    - Makes later track-level temporal attention possible if needed.
- **Important constraint:** do not refill a temporarily empty slot with a different track. Slot identity must remain fixed after the global top-12 tracks are selected.

## 5. Expand DINO scene grid from 2×3 to 4×4

- **Problem:** the current DINO dense feature map (`24×24`) is pooled to only `2×3 = 6` scene cells. This is very coarse for spatial reasoning, especially `evasion_space`, where the distinction between free roadway, occupied lane, shoulder, road edge, barrier, and nearby vehicles may occur within the same coarse cell.
- **Decision:** change scene adaptive pooling from `2×3` to `4×4`.
- **New scene representation**
    
    ```
    DINO dense grid: 24 × 24 × 768
              ↓ adaptive average pooling
    scene grid:       4 × 4 × 768
              ↓ flatten
    16 spatial scene tokens
              +
    1 DINO CLS token
              =
    17 scene tokens per frame
    ```
    
- With `12` persistent object tokens from toggle #4, the per-frame spatial Transformer input becomes:
    
    $$
    1\;\text{CLS} + 16\;\text{scene cells} + 12\;\text{objects} = 29\;\text{tokens}
    $$
    
- `29` tokens per frame is still computationally small, while preserving substantially more spatial layout than the current `19`-token design (`1 + 6 + 12`).
- Replace the learned scene-position embedding accordingly: `scene_positions` should change from shape `[7,384]` to `[17,384]` so the model can distinguish the CLS token and each of the sixteen fixed spatial cells.
- **Attribute-head interaction:** toggle #3 should also use all `16` event-local scene cells instead of the old `6`. Therefore its event-conditioned spatial set becomes `16 scene + 12 object = 28` tokens.
- **Reason for choosing 4×4:** it gives materially finer road-layout resolution while keeping token count low enough that the per-frame spatial Transformer remains inexpensive.

## 6. Replace entry-censor augmentation with random event-preserving crop

- **Problem:** the current augmentation sometimes starts the training clip after ENTRY and before COLLISION. This creates asymmetric samples where ENTRY is absent but COLLISION remains, which is not the desired training condition for the joint four-target model.
- **Decision:** remove the `start ∈ [ENTRY+1, COLLISION]` censor augmentation entirely. Every training crop must preserve **both ENTRY and COLLISION**.
- **Random event-preserving crop**
    - For a video of length `T`, with ground-truth event indices `e` and `c`, sample a random crop interval `[start, stop)` subject to:
        
        $$
        start \le e \le c < stop
        $$
        
    - Randomize available context before ENTRY and after COLLISION independently, instead of always using the complete sequence.
    - Conceptually:
        
        ```
        full video
        |-------------------- E -------- C ----------------------|
        
        possible crop A
               |------------- E -------- C -----------|
        
        possible crop B
        |-------------------- E -------- C ----|
        
        possible crop C
                      |------ E -------- C ------------------|
        
        All valid crops contain BOTH events.
        ```
        
    - A practical implementation can sample:
        
        ```python
        start = random integer in [0, entry]
        stop  = random integer in [collision + 1, T]
        ```
        
        optionally with minimum pre-ENTRY/post-COLLISION context constraints if experiments show they are needed.
        
- Re-index both labels after cropping:
    
    ```
    entry_index     = entry - start
    collision_index = collision - start
    ```
    
- Crop local DINO/object tensors to the same interval and retain only V-JEPA global tokens whose valid support lies inside the selected crop; recompute their crop-relative or physical timestamps consistently.
- `entry_supervised` should remain true for these crops because ENTRY is never intentionally removed.
- **Benefits**
    - Adds temporal-position and duration augmentation without deleting one of the supervised events.
    - Prevents the model from relying on ENTRY/COLLISION always occurring at a fixed relative location in the sequence.
    - Varies the amount of pre-event and post-event context seen during training.
    - Keeps training aligned with the intended joint task: localize both ENTRY and COLLISION and classify their associated attributes.

## 7. Online frozen V-JEPA 2.1 without photometric augmentation

- **Decision:** stop using precomputed V-JEPA feature caches during joint-model training. Keep V-JEPA 2.1 frozen and run it online after the random event-preserving temporal crop from toggle #6.
- **Motivation**
    - Cached V-JEPA features are difficult to reconcile cleanly with random temporal crops because each cached token was produced from a fixed temporal context in the original video.
    - Online V-JEPA guarantees that the global branch only sees frames inside the current crop, eliminating temporal-context leakage and removing the need for `global_support` bookkeeping or fallback logic.
    - It also allows crop-specific temporal sampling while keeping the existing index-only sampling policy.
- **Training pipeline**
    
    ```
    Offline cached once:
      RF-DETR / depth / tracking / geometry
      DINO scene + ROI features
    
    Each training iteration:
      load cached local features
            ↓
      random crop preserving ENTRY + COLLISION
            ↓
      slice cached DINO / geometry tensors to crop
            ↓
      load RGB frames belonging to crop
            ↓
      sample 16-frame V-JEPA clips at stride 4 (no image augmentation)
            ↓
      V-JEPA 2.1 forward under torch.inference_mode()
            ↓
      spatially pool V-JEPA tokens to global temporal tokens
            ↓
      trainable 1024→384 projection
            ↓
      local/global fusion + temporal refinement + heads
    ```
    
- **No gradient through V-JEPA**
    - Keep all V-JEPA parameters frozen and use evaluation mode.
    - Run the forward pass inside `torch.inference_mode()` so no autograd graph or backward activations are stored.
    
    ```python
    vjepa.eval()
    
    with torch.inference_mode():
        global_features = vjepa(video)
    
    # Training graph starts after the frozen features.
    global_features = global_features.clone().detach()  # outside inference_mode
    global_tokens = global_projection(global_features)
    ```
    
    - The trainable `1024 → 384` global projection and every downstream module still receive gradients normally.
    - This adds V-JEPA forward-compute cost each iteration, but avoids V-JEPA backward compute and activation-memory cost.
- **No photometric augmentation**
    - Do **not** apply brightness, contrast, saturation, gamma, hue, noise, JPEG, blur, or other stochastic photometric transforms to the online V-JEPA RGB input.
    - Reason: the DINO local branch remains cached from the original clean frames. Augmenting only V-JEPA would create a cross-branch appearance mismatch (`clean DINO` versus `augmented V-JEPA`) and could encourage the learned local/global fusion to discount the V-JEPA representation rather than learn complementary information.
    - Training and inference should therefore feed the same original RGB appearance into the frozen V-JEPA preprocessing path: temporal sampling → `384×384` letterbox → ImageNet normalization → V-JEPA forward.
    - Temporal cropping remains the intended online augmentation; do not introduce independent geometric or photometric image augmentation in this version.
- **Inference**
    - Run the same frozen V-JEPA path online using the original unaugmented RGB appearance, identical to training.
    - Use the same deterministic index-only sampling and `torch.inference_mode()`.
- **Expected benefits**
    - Exact compatibility with random ENTRY/COLLISION-preserving temporal crops.
    - No V-JEPA feature leakage from outside the crop.
    - Eliminates `global_support` filtering complexity.
    - Requires no FPS for V-JEPA sampling or inference.
    - Leaves open a future path to V-JEPA LoRA finetuning without redesigning the data pipeline.

## 8. W&B logging redesign aligned with competition metrics

- **Goal:** separate optimization diagnostics from competition-facing performance metrics, and make W&B report the same quantities that matter at inference and on the leaderboard.
- **Train loss metrics — log every 10 successful optimizer updates**
    - `train/loss`: total weighted training objective.
    - `train/loss_entry`: complete ENTRY event loss actually used for optimization, including the soft-distribution term and its CDF component.
    - `train/loss_collision`: complete COLLISION event loss actually used for optimization, including the soft-distribution term and its CDF component.
    - `train/loss_entry_side`: entry-side classification loss.
    - `train/loss_evasion_space`: evasion-space classification loss.
    - `train/loss_invalid_order`: ENTRY/COLLISION invalid-order penalty.
    - Do not expose `entry_cdf` and `collision_cdf` as primary W&B curves unless they are needed for debugging; `loss_entry` and `loss_collision` should represent the actual complete losses entering the objective.
- **Competition-facing train metrics — log only at the same interval as validation**
    - Do **not** log these every 10 optimizer updates. Compute and log them whenever validation is run (`val_every` / validation logging interval), using the complete train-epoch accumulation for that validation interval.
    - `train/acc_entry_0.3s`
    - `train/acc_collision_0.3s`
    - `train/f1_entry_side_macro`
    - `train/f1_evasion_space_macro`
    - `train/competition_score`
- **Validation metrics — log at every validation interval**
    - `val/loss`
    - `val/loss_entry`
    - `val/loss_collision`
    - `val/loss_entry_side`
    - `val/loss_evasion_space`
    - `val/loss_invalid_order`
    - `val/acc_entry_0.3s`
    - `val/acc_collision_0.3s`
    - `val/f1_entry_side_macro`
    - `val/f1_evasion_space_macro`
    - `val/competition_score`
- **Training-configuration metrics**
    - `train_config/epoch`
    - `train_config/lr_lora`
    - `train_config/lr_new_parameters`
    - These should be logged at the validation interval together with the competition-facing train/validation summary. Learning-rate logging may additionally be kept at the 10-update loss interval if needed for scheduler debugging, but the primary dashboard values should use the `train_config/*` namespace.
- **ENTRY/COLLISION accuracy definition**
    - Use the same constrained decoding procedure as inference so logged metrics match actual test-time behavior:
        
        $$
        (\hat e,\hat c)=\arg\max_{e\le c}\left[L_E(e)+L_C(c)\right]
        $$
        
    - Convert predicted and GT positions to physical time using the per-sample FPS information already represented by `frame_seconds`.
    - ENTRY is correct when:
        
        $$
        |t_{\hat e}-t_e|\le0.3\text{ s}
        $$
        
    - COLLISION is correct when:
        
        $$
        |t_{\hat c}-t_c|\le0.3\text{ s}
        $$
        
    - This metric is therefore FPS-aware: the same `±0.3 s` tolerance is used for 15 FPS, 30 FPS, or other valid source rates rather than using a fixed frame-count tolerance.
- **Macro-F1 calculation**
    - `entry_side`: compute F1 separately for LEFT and RIGHT, then average the two class F1 values.
    - `evasion_space`: compute F1 separately for class `0` and class `1`, then average them.
    - Do **not** calculate macro-F1 independently for each mini-batch and average the batch F1 scores.
    - Accumulate a `2×2` confusion matrix over the complete train/validation metric interval, reduce the counts across distributed workers if applicable, then calculate macro-F1 once from the aggregated counts.
- **Competition score**
    - Use the four competition-facing task metrics with the competition coefficients:
        
        $$
        S=0.35A_E+0.35A_C+0.15F1_{side}+0.15F1_{evasion}
        $$
        
    - `A_E`: ENTRY accuracy within `±0.3 s`.
    - `A_C`: COLLISION accuracy within `±0.3 s`.
    - `F1_side`: entry-side macro-F1.
    - `F1_evasion`: evasion-space macro-F1.
    - Log this as `train/competition_score` and `val/competition_score`.
- **Best-checkpoint selection**
    - If this local competition-score implementation exactly matches the official Stage 2 evaluation, choose `best.pt` using maximum `val/competition_score`, not minimum validation loss.
    - Validation loss should remain available for optimization diagnostics, but model selection should follow the leaderboard-aligned metric.
- **Desired directions**
    
    ```
    loss metrics                    ↓ lower is better
    invalid-order loss              ↓ lower is better
    ENTRY/COLLISION ±0.3s accuracy  ↑ higher is better
    entry-side macro-F1             ↑ higher is better
    evasion-space macro-F1          ↑ higher is better
    competition score               ↑ higher is better
    ```
    
- **Logging cadence summary**
    
    ```
    Every 10 optimizer updates:
      train/loss
      train/loss_entry
      train/loss_collision
      train/loss_entry_side
      train/loss_evasion_space
      train/loss_invalid_order
    
    At each validation interval (`val_every`):
      train competition-facing metrics
      all validation losses
      all validation competition-facing metrics
      train_config/epoch
      train_config/lr_lora
      train_config/lr_new_parameters
    ```
    

## 9. Training objective refinement

- **Current joint objective**
    - ENTRY and COLLISION are trained as temporal probability distributions rather than hard one-hot frame classification.
    - For each event, construct a Gaussian soft target in physical time:
        
        $$
        q_t \propto \exp\left(-\frac{(t-t^*)^2}{2\sigma^2}\right)
        $$
        
    - Current implementation uses `sigma_seconds = 0.1` for both ENTRY and COLLISION.
    - Event distribution loss:
        
        $$
        L_{dist}=-\sum_t q_t\log p_t
        $$
        
    - Add a temporal CDF-distance term so predictions far from the target are penalized more strongly than nearby errors:
        
        $$
        L_{event}=L_{dist}+0.05L_{CDF}
        $$
        
    - ENTRY-side uses cross entropy and evasion-space uses binary cross entropy with logits.
    - An ordering regularizer penalizes probability mass where COLLISION occurs before ENTRY:
        
        $$
        L_{order}=\sum_t p_E(t)P(C<t)
        $$
        
    - Current total objective:
        
        $$
        L=0.35L_E+0.35L_C+0.15L_{side}+0.15L_{evasion}+0.05L_{order}
        $$
        
- **Assessment**
    - Keep the overall formulation. The Gaussian temporal target is preferable to a hard one-hot frame label because adjacent-frame mistakes should not be treated like errors several seconds away.
    - Using `frame_seconds` makes the event objective FPS-aware.
    - The CDF term is useful because ordinary cross entropy does not explicitly encode temporal distance.
    - Keep the ordering penalty relatively weak; exact `ENTRY <= COLLISION` is already enforced by constrained decoding at inference.
    - Do not directly optimize the discrete competition metrics (`±0.3 s` accuracy or macro-F1). Keep them as evaluation metrics and use differentiable surrogate losses for training.
- **Modification 1: use different Gaussian widths for ENTRY and COLLISION**
    - COLLISION is usually visually sharper and more precisely annotatable than ENTRY.
    - ENTRY is semantically more ambiguous because the exact moment a vehicle enters the ego lane/path can vary slightly by annotation.
    - Recommended initial values:
        
        $$
        \sigma_{collision}=0.10\text{ s}
        $$
        
        $$
        \sigma_{entry}=0.15\text{--}0.20\text{ s}
        $$
        
    - The competition metric remains unchanged at `±0.3 s`; this only changes label smoothing during optimization.
- **Modification 2: do not assume competition coefficients imply balanced optimization gradients**
    - Keep `0.35 / 0.35 / 0.15 / 0.15` as the initial task weights so architecture changes and loss-weight changes are not confounded in the same experiment.
    - However, event cross-entropy losses can naturally have a much larger numerical scale than binary classification losses. Therefore the effective gradient contribution may be much more localization-heavy than the nominal coefficients suggest.
    - After the new W&B metrics are implemented, inspect actual loss magnitudes and, for a short diagnostic run, optionally inspect gradient norms on shared parameters:
        
        $$
        									\|\nabla_\theta L_E\|,\quad
        \|\nabla_\theta L_C\|,\quad
        \|\nabla_\theta L_{side}\|,\quad
        \|\nabla_\theta L_{evasion}\|
        $$
        
    - Only adjust task weights if the logged behavior shows that side/evasion learning is being dominated by localization.
- **Interaction with temporal crop modification**
    - Once toggle #6 is implemented, every training crop must preserve both ENTRY and COLLISION.
    - The current `entry_supervised` censoring path can therefore be removed for normal training, and every sample can contribute to all four supervised targets.
- **Recommended final structure**
    
    ```
    ENTRY:
    Gaussian temporal CE + 0.05 × temporal CDF loss
    sigma ≈ 0.15–0.20 s
    
    COLLISION:
    Gaussian temporal CE + 0.05 × temporal CDF loss
    sigma = 0.10 s
    
    ENTRY SIDE:
    Cross Entropy
    
    EVASION SPACE:
    BCEWithLogits
    
    STRUCTURAL REGULARIZER:
    0.05 × invalid-order penalty
    
    INITIAL TOTAL:
    0.35 × ENTRY
    + 0.35 × COLLISION
    + 0.15 × ENTRY SIDE
    + 0.15 × EVASION SPACE
    + 0.05 × INVALID ORDER
    ```