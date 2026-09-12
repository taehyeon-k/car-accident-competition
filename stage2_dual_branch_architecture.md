# Stage 2 Architecture Specification
## Dual-Branch Local–Global Precise Event Spotting for Car-Accident Understanding

**Status:** Proposed Stage 2 v2 architecture  
**Purpose:** Replace the current coarse-to-fine cascade with a single end-to-end prediction pipeline that combines native-frame local detail with sparse global video context.  
**Primary targets:** `entry_frame`, `collision_frame`, `entry_side`, `evasion_space`  
**Core backbones:** frozen DINOv3 ViT-B/16 local branch + trainable V-JEPA 2.1 ViT-L/16 global branch with LoRA  
**Key principle:** **the local branch owns temporal precision; the global branch provides context.**

---

# 1. Executive Summary

The previous Stage 2 architecture used a two-stage coarse-to-fine pipeline:

1. a coarse model predicted approximate temporal bins for ENTRY and COLLISION;
2. a fine model received a small native-frame window around the predicted bin and localized the exact frame.

That design is reasonable, but it introduces a structural failure mode: if the coarse model routes the fine model to the wrong temporal region, the fine model may never see the true event. Training also differs from inference because fine training is predominantly conditioned on windows known to contain the ground-truth event, whereas inference uses coarse-model predictions.

The proposed architecture removes that hard routing entirely.

Instead, every native frame receives a high-resolution local representation from **frozen DINOv3-B**, while a sparse subset of frames is processed jointly by **V-JEPA 2.1 ViT-L with LoRA** to produce long-range driving context. The two streams are fused through **gated cross-attention**, preserving the native-frame DINO representation as the identity path. A lightweight full-resolution temporal refinement module models short-range state transitions. ENTRY and COLLISION are then predicted directly as probability distributions over **all native frames**.

At a high level:

```text
                               FULL VIDEO
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
        LOCAL / NATIVE STREAM                GLOBAL / CONTEXT STREAM
        every native frame                   sparse 32-frame sample
                 │                                   │
          frozen DINOv3-B                    V-JEPA 2.1 ViT-L
           dense patches                         + LoRA
                 │                                   │
      scene + vehicle ROI features        16 tubelet representations
                 │                                   │
      cached detector/depth geometry      spatially pooled + projected
                 │                                   │
          spatial interaction                      G
                 │                                   │
                 ▼                                   │
             L ∈ R^(T×384)                          │
                 │                                   │
        local temporal refinement                   │
                 │ Q                                 │ K,V
                 └──────────► gated cross-attention ◄┘
                                   │
                                   ▼
                              F ∈ R^(T×384)
                                   │
                    2–4 full-resolution temporal blocks
                                   │
                                   ▼
                              H ∈ R^(T×384)
                         ┌─────────┴─────────┐
                         ▼                   ▼
                    ENTRY query        COLLISION query
                         │                   │
                         ▼                   ▼
                    p_E(1:T)            p_C(1:T)
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                        constrained E ≤ C decoding
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
          entry_frame        collision_frame     event-conditioned
                                                    attribute heads
                                                ┌────────┴────────┐
                                                ▼                 ▼
                                           entry_side       evasion_space
```

The model is therefore **single-stage with respect to prediction**: there is no proposal window, no temporal routing, and no second model that can be deprived of the correct event.

---

# 2. Problem Definition

For each usable accident video, let the native decoded frame sequence be

\[
X = \{x_t\}_{t=0}^{T-1},
\]

where \(T\) varies across videos.

The model predicts four labels:

1. **ENTRY frame**
   \[
   t_E \in \{0,\ldots,T-1\}
   \]

2. **COLLISION frame**
   \[
   t_C \in \{0,\ldots,T-1\}
   \]

3. **ENTRY side**
   \[
   y_{\text{side}} \in \{\text{LEFT},\text{RIGHT}\}
   \]

4. **Evasion space**
   \[
   y_{\text{evasion}} \in \{0,1\}
   \]

For valid samples, the accident chronology should normally satisfy

\[
t_E \le t_C.
\]

The primary localization challenge is not merely action recognition. It is **precise event spotting**: the output is a single exact frame for ENTRY and a single exact frame for COLLISION.

This distinction strongly influences the architecture. Temporal downsampling is acceptable internally, but the final localization representation should remain or return to native-frame resolution.

---

# 3. Design Goals

The architecture is designed around the following requirements.

## 3.1 Exact native-frame localization

ENTRY and COLLISION must be predicted at native-frame resolution. A representation that permanently compresses every two, four, or eight frames into one timestamp is undesirable unless another branch restores exact frame information.

## 3.2 Full-video visibility

Every candidate event frame should remain available to the localization head. The architecture must avoid catastrophic errors caused by a hard coarse temporal router.

## 3.3 Long-range accident context

Some frames are locally ambiguous. Correct interpretation may require knowing the broader progression:

```text
normal driving
    ↓
other vehicle approaches
    ↓
other vehicle enters ego trajectory
    ↓
evasion / interaction
    ↓
collision
```

A global video encoder is useful for this role.

## 3.4 Fine spatial/object detail

ENTRY and COLLISION often depend on vehicle-level details that may be small in the image. Therefore the local representation should retain:

- scene appearance;
- individual tracked vehicle appearance;
- vehicle positions and sizes;
- relative depth;
- temporal geometric trends.

## 3.5 Small-data robustness

The labeled usable dataset is relatively small. The design should avoid unnecessary end-to-end fine-tuning of very large backbones when strong frozen representations can be used.

## 3.6 Practical training cost

DINOv3 should not be recomputed unnecessarily on every epoch if it remains frozen. The live trainable video backbone should be limited to the sparse V-JEPA global branch.

---

# 4. Core Architectural Principle

The model intentionally assigns different responsibilities to the two visual branches.

## Local branch: DINOv3-B

Responsible for:

- exact native-frame appearance;
- high-resolution spatial semantics;
- vehicle ROI appearance;
- preserving frame-to-frame distinctions;
- supporting exact ENTRY/COLLISION localization.

It is **frozen** and uses **no LoRA**.

## Global branch: V-JEPA 2.1 ViT-L

Responsible for:

- broad temporal context;
- global driving-state evolution;
- relationships across distant moments;
- helping ambiguous local frames interpret where they lie in the accident progression.

It is trained with **LoRA** and may optionally retain the already-established policy of unfreezing the final few blocks if later experiments justify it, but the initial specification assumes LoRA is the primary adaptation mechanism.

## Fusion rule

The global branch should **condition** the local branch.

It should not replace the local representation.

Formally:

\[
\boxed{\text{local stream owns temporal precision}}
\]

\[
\boxed{\text{global stream supplies contextual evidence}}
\]

---

# 5. Input and Preprocessing

## 5.1 Native video frames

Decode each video into its original frame sequence.

For a representative example:

\[
T = 500.
\]

The architecture should support variable \(T\).

Do not make 500 a model constant.

## 5.2 Detector and depth observations

Reuse the existing Stage 2 geometry pipeline:

- RF-DETR Small;
- Depth Anything V2 Small;
- cached per-native-frame observations.

For each detected/tracked object, retain the existing geometry channels:

1. normalized center \(x\);
2. normalized bottom \(y\);
3. normalized width;
4. normalized height;
5. normalized area;
6. relative depth proximity;
7. within-frame depth rank;
8. log-area delta;
9. recent log-area slope.

Thus each object has

\[
g^{raw}_{t,k} \in \mathbb{R}^{9}.
\]

The detector and depth networks remain frozen.

## 5.3 Tracking

Because the new architecture operates over the full native timeline instead of isolated fine windows, tracking should be performed over the full video or over sufficiently overlapping chunks with stable ID reconciliation.

The implementation may reuse the project's current matching logic, but the output should be a consistent set of object trajectories across the native video.

Keep at most

\[
K = 12
\]

object tracks per frame, matching the current model design.

Use a validity mask for missing/padded objects.

---

# 6. Local Branch: Frozen DINOv3-B

## 6.1 Backbone

Use:

**DINOv3 ViT-B/16**

with:

- no LoRA;
- no gradient updates;
- inference mode;
- BF16 where supported.

For 384×384 input with patch size 16:

\[
24 \times 24 = 576
\]

patch positions per frame.

The dense representation for frame \(t\) is approximately

\[
P_t \in \mathbb{R}^{24 \times 24 \times 768}.
\]

For a \(T=500\) video:

\[
P \in \mathbb{R}^{500 \times 24 \times 24 \times 768}.
\]

Do **not** feed this entire tensor directly into the temporal model.

Its role is to provide high-quality spatial features that are immediately compressed into task-specific scene/object representations.

---

# 7. DINO Feature Extraction Batching

The proposed implementation may process native frames in large image batches, for example:

```text
frames 0–127    → DINO
frames 128–255  → DINO
frames 256–383  → DINO
frames 384–499  → DINO
```

However, **128 is not an architectural requirement**.

Use the largest batch size that fits efficiently on the target GPU.

A typical extraction loop should use:

```python
model.eval()

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ...
```

Possible batch sizes include 32, 64, 96, 128, etc.

The batch dimension is independent across frames: this does not mean DINO is performing temporal modeling.

---

# 8. DINO Spatial Compression

The dense DINO grid should immediately be converted into:

1. one scene/global token;
2. up to 12 vehicle ROI tokens.

## 8.1 Scene token

Spatially average or attention-pool the DINO patch grid:

\[
s_t =
\operatorname{Pool}_{xy}(P_t)
\in \mathbb{R}^{768}.
\]

Project:

\[
\tilde s_t = W_s s_t
\in \mathbb{R}^{384}.
\]

This token represents the whole native frame.

## 8.2 Vehicle ROI appearance

For tracked vehicle \(k\) with box \(b_{t,k}\):

\[
a_{t,k}
=
\operatorname{ROIAlign}(P_t,b_{t,k}).
\]

Pool the ROI to a vector:

\[
a_{t,k}\in\mathbb{R}^{768}.
\]

Project:

\[
\tilde a_{t,k}
=
W_a a_{t,k}
\in\mathbb{R}^{256}.
\]

ROIAlign should use DINO patch-coordinate-aligned boxes, taking into account resizing/cropping.

## 8.3 Geometry embedding

Encode the 9-D geometry vector:

\[
e^g_{t,k}
=
MLP_g(g^{raw}_{t,k})
\in\mathbb{R}^{128}.
\]

## 8.4 Object token

Concatenate appearance and geometry:

\[
o_{t,k}
=
[\tilde a_{t,k};e^g_{t,k}]
\in\mathbb{R}^{384}.
\]

So every object token remains dimensionally compatible with the scene token.

---

# 9. Per-Frame Spatial Interaction

Construct a token set for every native frame:

\[
Z_t =
[
\tilde s_t,
o_{t,1},
\ldots,
o_{t,K}
].
\]

With \(K=12\):

\[
Z_t \in \mathbb{R}^{13 \times 384}.
\]

Use a small spatial interaction network, for example:

- 2 Transformer encoder blocks;
- 6 attention heads;
- MLP ratio around 4;
- object validity mask;
- residual connections;
- LayerNorm.

The first token acts as the scene/global token and gathers object interaction information.

After the spatial blocks, extract the updated first token:

\[
L_t^{(0)} \in \mathbb{R}^{384}.
\]

Across the full video:

\[
L^{(0)}
=
[L_0^{(0)},\ldots,L_{T-1}^{(0)}]
\in\mathbb{R}^{T\times384}.
\]

This is the compact native-frame local sequence.

For \(T=500\):

\[
L^{(0)} \in \mathbb{R}^{500\times384}.
\]

---

# 10. Optional Explicit Local Difference Features

DINO itself is applied frame-by-frame. Therefore precise event localization benefits from explicitly exposing changes between adjacent native-frame representations.

Define:

\[
\Delta_t = L_t^{(0)} - L_{t-1}^{(0)}
\]

and optionally the second temporal difference:

\[
A_t =
L_t^{(0)}
-
2L_{t-1}^{(0)}
+
L_{t-2}^{(0)}.
\]

Construct:

\[
\bar L_t
=
MLP_\Delta
(
[
L_t^{(0)};
\Delta_t;
A_t
]
).
\]

Use a residual:

\[
L_t^{(1)}
=
L_t^{(0)}
+
\bar L_t.
\]

This is **not** a separate optical-flow or motion branch.

It is simply temporal processing of the already-existing local representation.

The purpose is to make transitions such as:

```text
approach → entry
near-contact → contact
```

more linearly accessible.

This module should be ablated rather than assumed mandatory.

---

# 11. Local Temporal Refinement Before Fusion

Before asking the global V-JEPA branch for context, the local sequence may receive a small amount of native-resolution temporal processing.

Recommended starting point:

- residual temporal Conv1D, kernel 3;
- residual temporal Conv1D, kernel 5;
- optionally 1–2 full-resolution Transformer/SGP-style blocks.

For example:

\[
L^{local}
=
TemporalLocal(L^{(1)}).
\]

Shape is unchanged:

\[
L^{local}\in\mathbb{R}^{T\times384}.
\]

This module's job is **short-range temporal discrimination**, not long-range context.

The V-JEPA branch already handles broad global context.

---

# 12. DINO Caching Strategy

Because DINOv3-B is frozen, recomputing it during every training epoch is unnecessary.

## 12.1 Recommended cache level

Do not cache the full raw DINO grid unless ROI definitions are still changing frequently.

Instead cache compact outputs such as:

- scene vector \(s_t\);
- ROI appearance vectors \(a_{t,k}\);
- ROI validity masks;
- track IDs / box associations.

A representative compact cache per frame might contain:

\[
768 + 12\times768
\]

values before trainable projection.

Alternatively, if the appearance projections are also kept frozen after an initial design decision, cache their lower-dimensional outputs:

\[
384 + 12\times256.
\]

The latter is much smaller but less flexible.

## 12.2 Recommended compromise

Cache:

\[
\text{scene DINO vector: } 768
\]

and

\[
\text{object ROI DINO vectors: } K\times768.
\]

Keep the projection layers trainable.

This allows the model to learn task-specific compression without rerunning DINO.

## 12.3 Full dense cache

At 384×384 and BF16, one full DINO dense grid is approximately:

\[
24\times24\times768\times2\text{ bytes}
\approx 0.885\text{ MB/frame}.
\]

For 500 frames:

\[
\approx443\text{ MB/video}.
\]

That is unnecessary for ordinary training once ROI extraction is fixed.

---

# 13. Photometric Augmentation with Cached DINO

Caching DINO features changes the augmentation policy.

If image-space photometric augmentation is

\[
\tilde x = Aug_\theta(x)
\]

then in general:

\[
DINO(\tilde x)
\neq
Aug'(DINO(x)).
\]

Therefore a single clean DINO cache cannot exactly reproduce random online brightness/color/blur augmentations.

## Recommended policy

Use:

\[
\boxed{\text{clean frozen DINO cache + feature-level regularization}}
\]

for the local branch.

Possible feature-level augmentation:

- mild Gaussian feature noise;
- channel dropout;
- token dropout;
- ROI-object dropout;
- temporal frame/token dropout;
- feature scaling jitter.

## Optional multi-view cache

If camera/photometric robustness becomes a measured problem, precompute 2–3 temporally consistent DINO views:

- original;
- mild brightness/contrast/color;
- mild blur/compression-like perturbation.

Randomly choose one cached view during training.

Do not independently randomize photometric parameters for neighboring frames because artificial frame-to-frame brightness jumps can become false event cues.

---

# 14. Global Branch: V-JEPA 2.1 ViT-L

## 14.1 Input frame sampling

Initial setting:

\[
N_G = 32
\]

frames sampled across the complete video.

Use stratified sampling.

Divide the normalized timeline into 32 bins and select one frame per bin.

During training, add small jitter within each bin.

At validation/inference, use deterministic bin centers.

Let the sampled native-frame indices be:

\[
I = \{i_0,\ldots,i_{31}\}.
\]

For \(T=500\), these cover the full accident at roughly uniform temporal spacing.

## 14.2 Why 32 initially

32 provides:

- low global compute;
- full-video coverage;
- enough V-JEPA temporal structure for a first model;
- 16 tubelets with tubelet size 2.

A key ablation is:

\[
32 \text{ vs } 64
\]

V-JEPA input frames.

64 input frames would produce 32 tubelet tokens and may improve global accident evolution modeling.

---

# 15. V-JEPA Backbone and Adaptation

Use:

**V-JEPA 2.1 ViT-L/16**

with:

- 384×384 input;
- feature dimension 1024;
- tubelet size 2;
- LoRA adaptation;
- BF16 training.

For 32 input frames:

\[
32 / 2 = 16
\]

temporal tubelets.

The dense V-JEPA output is expected to be approximately:

\[
V^{dense}
\in
\mathbb{R}^{16\times24\times24\times1024}.
\]

Do not keep this full dense output beyond the global-branch forward pass.

---

# 16. V-JEPA Global Token Extraction

Spatially pool each tubelet:

\[
g_j
=
\operatorname{Pool}_{xy}
(
V^{dense}_j
)
\in\mathbb{R}^{1024},
\qquad
j=0,\ldots,15.
\]

Project:

\[
G_j
=
W_G g_j
\in\mathbb{R}^{384}.
\]

Then:

\[
G
=
[G_0,\ldots,G_{15}]
\in\mathbb{R}^{16\times384}.
\]

These are more accurately described as:

**spatially pooled V-JEPA tubelet tokens**

rather than conventional CLS tokens.

---

# 17. Global Token Native-Time Coordinates

The global tokens correspond to sampled native-frame locations.

Do not represent them merely by indices \(0,\ldots,15\).

For V-JEPA tubelet \(j\), associated with sampled native indices \(i_{2j}\) and \(i_{2j+1}\), define:

\[
\tau^G_j
=
\frac{
(i_{2j}+i_{2j+1})/2
}{
T-1
}.
\]

Thus:

\[
\tau^G_j \in [0,1].
\]

Likewise each local frame has normalized time:

\[
\tau^L_t
=
\frac{t}{T-1}.
\]

These timestamps are used in relative temporal attention bias.

---

# 18. Online Augmentation for V-JEPA

Unlike DINO, V-JEPA is trainable with LoRA.

Therefore the V-JEPA RGB input remains online during training.

Photometric augmentation may include:

- brightness;
- contrast;
- color jitter;
- mild blur;
- mild compression-style degradation.

Critically, apply the **same photometric parameters to all 32 sampled frames in the clip**.

Do not independently jitter every sampled frame.

Otherwise augmentation-induced appearance jumps may be learned as temporal events.

---

# 19. Why Direct Concatenation Is Not Preferred

A naive strategy could upsample the 16 V-JEPA tokens to \(T\) frames and concatenate them with DINO:

\[
F_t = [L_t;Upsample(G)_t].
\]

This is not preferred because:

1. interpolation turns each sparse V-JEPA token into many nearly identical frame-level context vectors;
2. this may blur native-frame boundaries;
3. it ignores semantic relevance: not every local frame should receive the same neighboring global token;
4. there is no learned retrieval mechanism between local appearance and global driving state.

Instead use cross-attention.

---

# 20. Preferred Fusion: Local-to-Global Cross-Attention

Let:

\[
L \equiv L^{local}
\in
\mathbb{R}^{T\times384}
\]

and

\[
G
\in
\mathbb{R}^{N_g\times384}
\]

where initially \(N_g=16\).

Use the local native-frame tokens as **queries**.

Use V-JEPA tokens as **keys and values**:

\[
Q=W_Q L
\]

\[
K=W_K G
\]

\[
V=W_V G.
\]

Attention logits are:

\[
a_{tj}
=
\frac{
q_t^\top k_j
}{
\sqrt d
}.
\]

Then:

\[
A_{tj}
=
softmax_j(a_{tj}).
\]

Global context retrieved by native frame \(t\):

\[
C_t
=
\sum_{j=0}^{N_g-1}
A_{tj}v_j.
\]

Thus:

\[
C
\in
\mathbb{R}^{T\times384}.
\]

For \(T=500\) and \(N_g=16\), cross-attention has only:

\[
500\times16=8000
\]

query-key pairings per attention head.

This is computationally trivial relative to either backbone.

---

# 21. Important Interpretation of Q, K, and V

Cross-attention:

\[
C=Attn(L,G,G)
\]

does **not** mean \(C\) automatically contains the complete DINO representation.

The output is:

\[
C_t
=
\sum_jA_{tj}W_VG_j.
\]

The local DINO token \(L_t\) affects the **attention weights**, but the output content is a weighted mixture of **V-JEPA value vectors**.

Therefore:

\[
C_t
\]

should be interpreted as:

> global V-JEPA context retrieved specifically for local native frame \(t\).

It is not a replacement for \(L_t\).

This is why the local identity path must be preserved.

---

# 22. Relative Temporal Bias in Fusion

Use actual normalized native timestamps.

For local frame \(t\) and V-JEPA global token \(j\):

\[
\Delta\tau_{tj}
=
\tau^L_t-\tau^G_j.
\]

Generate an attention bias:

\[
b_{tj}
=
MLP_{time}(\Delta\tau_{tj})
\]

or use a learned relative-position embedding/bucket.

Then:

\[
A_{tj}
=
softmax_j
\left(
\frac{q_t^\top k_j}{\sqrt d}
+
b_{tj}
\right).
\]

This allows global attention to reason both semantically and temporally.

A native frame is free to attend to distant global states, but the model knows how far away those states are.

---

# 23. Preferred Gated Residual Fusion

Do not simply replace \(L_t\) with \(C_t\).

Do not assume:

\[
F_t=C_t.
\]

The recommended fusion is:

\[
\gamma_t
=
\sigma
\left(
MLP_{gate}([L_t;C_t])
\right).
\]

Possible gate forms:

### Scalar gate

\[
\gamma_t\in[0,1].
\]

### Channel-wise gate

\[
\gamma_t\in[0,1]^{384}.
\]

Channel-wise gating is more expressive but may overfit more easily.

Initial implementation should use either:

- scalar per frame; or
- small grouped/channel gate.

Project global context:

\[
\tilde C_t=P_C(C_t).
\]

Fuse:

\[
\boxed{
F_t
=
L_t
+
\gamma_t\odot \tilde C_t
}
\]

with:

\[
F
\in
\mathbb{R}^{T\times384}.
\]

This preserves exact local information while allowing V-JEPA to modify it when global context is useful.

---

# 24. Gate Initialization

Initialize the global-context gate conservatively.

For a sigmoid gate with bias:

\[
b_{gate}\approx-2
\]

gives:

\[
\sigma(-2)\approx0.12.
\]

Thus at the start of training:

\[
F_t
\approx
L_t + 0.12\tilde C_t.
\]

This is desirable.

Initially, the model behaves mostly as a native-frame local model.

Training can increase the V-JEPA contribution only where it is useful.

---

# 25. Optional Local Self-Attention Before Global Cross-Attention

An optional stronger fusion stack is:

1. local temporal self-attention;
2. global cross-attention;
3. gated residual fusion.

Formally:

\[
L'
=
Attn(L,L,L)
\]

followed by:

\[
C
=
Attn(L',G,G).
\]

Then:

\[
F
=
L'
+
Gate(L',C)\odot P(C).
\]

Interpretation:

### Local self-attention

\[
Q=K=V=L
\]

asks:

> how do native frames relate to one another?

### Global cross-attention

\[
Q=L', K=V=G
\]

asks:

> which global accident states help interpret this exact native frame?

These are different operations and should not be collapsed into a mathematically inconsistent Q/K/V assignment.

This self-attention block is optional because the subsequent temporal refinement network already provides local communication.

---

# 26. Temporal Refinement After Fusion

The latest design does **not** require a full temporal encoder-decoder by default.

V-JEPA already supplies broad global context.

The remaining purpose of temporal processing is primarily:

\[
\boxed{\text{native-frame transition reasoning}}
\]

such as:

- before ENTRY vs exact ENTRY vs after ENTRY;
- approach vs contact;
- adjacent-frame discriminability.

Therefore start with a lightweight full-resolution temporal network.

---

# 27. Recommended Full-Resolution Temporal Network

Input:

\[
F\in\mathbb{R}^{T\times384}.
\]

Output:

\[
H\in\mathbb{R}^{T\times384}.
\]

Recommended initial configuration:

### Option A: temporal conv + Transformer

```text
F
│
├─ residual Conv1D k=3
│
├─ residual Conv1D k=5
│
├─ Transformer block
│
├─ Transformer block
│
└─ H
```

### Option B: 2–4 SGP-style / discriminability-oriented blocks

This may be attractive because precise event spotting benefits from preventing adjacent temporal tokens from becoming over-smoothed.

### Option C: dilated temporal convolution blocks

For very small data, a convolutional temporal network may generalize better than a deeper Transformer.

All options preserve:

\[
T\rightarrow T.
\]

There is no required temporal downsampling.

---

# 28. Why a Full Encoder-Decoder Is Not Mandatory

A temporal encoder-decoder of the form:

\[
T\rightarrow T/2\rightarrow T/4\rightarrow T/8\rightarrow T
\]

can capture multiscale context while restoring frame resolution.

However, in this architecture:

- V-JEPA already provides long-range global context;
- DINO provides native-frame detail;
- the model is trained on a relatively small labeled set.

Therefore a deep temporal encoder-decoder may be redundant or over-parameterized.

The recommended experimental order is:

### Variant A

\[
Fusion \rightarrow EventHeads
\]

### Variant B — recommended starting model

\[
Fusion
\rightarrow
2\text{–}4\ FullResolutionTemporalBlocks
\rightarrow
EventHeads
\]

### Variant C

\[
Fusion
\rightarrow
TemporalEncoderDecoder
\rightarrow
EventHeads
\]

Only introduce Variant C if validation shows that long-range temporal modeling after fusion remains a bottleneck.

---

# 29. Event Localization Heads

ENTRY and COLLISION should be predicted directly over the complete native-frame sequence.

Do not regress a single continuous timestamp directly as the only signal.

Instead produce distributions over all native frames.

---

# 30. Event Query Representations

Create two trainable event vectors:

\[
q_E
\in\mathbb{R}^{384}
\]

for ENTRY and

\[
q_C
\in\mathbb{R}^{384}
\]

for COLLISION.

Optionally pass the two event queries through a tiny self-attention block so they can exchange information:

\[
[q_E',q_C']
=
EventQueryBlock([q_E,q_C]).
\]

This exposes the known semantic relationship between the two events.

---

# 31. Event Query Conditioning on Global Context

Optionally condition each event query on V-JEPA global tokens:

\[
\bar q_E
=
q_E'
+
Attn(q_E',G,G)
\]

\[
\bar q_C
=
q_C'
+
Attn(q_C',G,G).
\]

This gives the query itself an understanding of the overall accident evolution before it compares itself with native frames.

This is optional because the native-frame sequence is already V-JEPA-conditioned.

---

# 32. Frame Scoring

Project the final temporal features:

\[
k_t^H = W_H H_t.
\]

For ENTRY:

\[
s_E(t)
=
\frac{
(W_E \bar q_E)^\top k_t^H
}{
\sqrt d
}.
\]

For COLLISION:

\[
s_C(t)
=
\frac{
(W_C \bar q_C)^\top k_t^H
}{
\sqrt d
}.
\]

Then:

\[
p_E(t)
=
softmax_t(s_E(t))
\]

\[
p_C(t)
=
softmax_t(s_C(t)).
\]

Each distribution sums to 1:

\[
\sum_t p_E(t)=1
\]

\[
\sum_t p_C(t)=1.
\]

There are no coarse bins.

There are no candidate windows.

---

# 33. Gaussian Soft Localization Targets

A one-hot target treats all incorrect frames equally.

That is undesirable for precise temporal localization.

If the ground-truth ENTRY frame is \(t_E\), define:

\[
y_E(t)
=
\frac{
\exp
\left[
-\frac{(t-t_E)^2}{2\sigma^2}
\right]
}{
\sum_u
\exp
\left[
-\frac{(u-t_E)^2}{2\sigma^2}
\right]
}.
\]

Initial recommendation:

\[
\sigma=1.0
\]

native frame.

Ablate:

\[
\sigma \in \{0.75,1.0,1.5,2.0\}.
\]

The localization distribution loss is:

\[
L^{dist}_E
=
-\sum_t
y_E(t)
\log p_E(t).
\]

Likewise:

\[
L^{dist}_C.
\]

---

# 34. Expected-Position Loss

Convert each predicted distribution to a differentiable normalized expected timestamp.

Define normalized native time:

\[
r_t=\frac{t}{T-1}.
\]

Then:

\[
\hat r_E
=
\sum_t
r_t p_E(t).
\]

Ground truth:

\[
r_E=\frac{t_E}{T-1}.
\]

Use:

\[
L^{pos}_E
=
SmoothL1(\hat r_E,r_E).
\]

Likewise:

\[
L^{pos}_C.
\]

This encourages globally sensible localization even if the distribution is somewhat spread.

---

# 35. Before/After State Auxiliary Heads

Keep the strong idea already used by the current fine model.

ENTRY state target:

\[
y_E^{state}(t)
=
\mathbf{1}[t\ge t_E].
\]

COLLISION state target:

\[
y_C^{state}(t)
=
\mathbf{1}[t\ge t_C].
\]

Predict:

\[
\hat y_E^{state}(t)
=
\sigma(MLP_E^{state}(H_t))
\]

and:

\[
\hat y_C^{state}(t)
=
\sigma(MLP_C^{state}(H_t)).
\]

Use BCE:

\[
L_E^{state}
=
BCE(\hat y_E^{state},y_E^{state})
\]

\[
L_C^{state}
=
BCE(\hat y_C^{state},y_C^{state}).
\]

Why this matters:

The event timestamp gives only one strong labeled position.

The before/after target provides supervision to every native frame.

For a small dataset, this denser signal is valuable.

---

# 36. Event Ordering Loss

Use the known chronology:

\[
ENTRY \le COLLISION.
\]

Using expected normalized timestamps:

\[
L_{order}
=
ReLU(\hat r_E-\hat r_C)^2.
\]

This penalizes predictions in which expected ENTRY occurs after expected COLLISION.

Do not make this weight excessively large; constrained inference will enforce chronology exactly.

---

# 37. Constrained Joint Event Decoding

At inference, do not independently take:

\[
argmax(p_E)
\]

and:

\[
argmax(p_C)
\]

without considering event order.

Instead solve:

\[
(\hat t_E,\hat t_C)
=
\arg\max_{e\le c}
[
\log p_E(e)+\log p_C(c)
].
\]

This guarantees:

\[
\hat t_E\le\hat t_C.
\]

The optimization can be implemented in \(O(T)\) using cumulative best ENTRY scores while scanning collision positions.

---

# 38. Soft Event Embeddings

For downstream attribute prediction, use differentiable event-conditioned pooling rather than hard argmax during training.

ENTRY embedding:

\[
z_E
=
\sum_t
p_E(t)H_t.
\]

COLLISION embedding:

\[
z_C
=
\sum_t
p_C(t)H_t.
\]

Both are:

\[
z_E,z_C\in\mathbb{R}^{384}.
\]

These representations concentrate around the predicted event while allowing gradients to flow through localization probabilities.

---

# 39. Global Video Context Vector

Pool the V-JEPA sequence:

\[
z_G
=
AttentionPool(G)
\in\mathbb{R}^{384}.
\]

A simpler mean pool is acceptable for the initial baseline:

\[
z_G
=
\frac1{N_g}
\sum_jG_j.
\]

This vector represents broad video context for attribute heads.

---

# 40. ENTRY Side Head

`entry_side` is semantically tied to the ENTRY event.

Therefore do not predict it only from an unrelated global video pool.

Use:

\[
u_{side}
=
[z_E;z_G].
\]

Then:

\[
p_{side}
=
softmax(
MLP_{side}(u_{side})
).
\]

Output:

\[
p_{side}
\in
\mathbb{R}^2.
\]

Loss:

\[
L_{side}
=
CE(p_{side},y_{side}).
\]

This makes the classifier focus on the entering vehicle state while retaining broad scene orientation/context.

---

# 41. ENTRY-to-COLLISION Soft Interval

`evasion_space` describes behavior during the interaction interval rather than one isolated frame.

Use the predicted event distributions to construct a differentiable soft interval.

One implementation uses expected normalized positions mapped back to frame coordinates:

\[
\mu_E=(T-1)\hat r_E
\]

\[
\mu_C=(T-1)\hat r_C.
\]

Define:

\[
m_t
=
\sigma
\left(
\frac{t-\mu_E}{\tau}
\right)
\cdot
\sigma
\left(
\frac{\mu_C-t}{\tau}
\right).
\]

Here \(\tau\) controls boundary softness.

Then:

\[
z_{EC}
=
\frac{
\sum_tm_tH_t
}{
\sum_tm_t+\epsilon
}.
\]

This summarizes the predicted interaction interval.

---

# 42. Evasion-Space Head

Use:

\[
u_{evasion}
=
[z_E;z_{EC};z_C;z_G].
\]

Then:

\[
p_{evasion}
=
\sigma(
MLP_{evasion}(u_{evasion})
).
\]

Loss:

\[
L_{evasion}
=
BCE(p_{evasion},y_{evasion}).
\]

This is semantically preferable to whole-video pooling because it explicitly asks:

> what happened from ENTRY through COLLISION?

---

# 43. Total Objective

Define event localization losses:

\[
L_E
=
L_E^{dist}
+
\lambda_{pos}L_E^{pos}
+
\lambda_{state}L_E^{state}
\]

\[
L_C
=
L_C^{dist}
+
\lambda_{pos}L_C^{pos}
+
\lambda_{state}L_C^{state}.
\]

Then:

\[
L_{total}
=
\lambda_E L_E
+
\lambda_C L_C
+
\lambda_{side}L_{side}
+
\lambda_{evasion}L_{evasion}
+
\lambda_{order}L_{order}.
\]

## Suggested initial weights

A reasonable starting point:

\[
\lambda_E=1.0
\]

\[
\lambda_C=1.0
\]

\[
\lambda_{pos}=0.2
\]

\[
\lambda_{state}=0.2
\]

\[
\lambda_{side}=0.3
\]

\[
\lambda_{evasion}=0.3
\]

\[
\lambda_{order}=0.05.
\]

These are starting values, not fixed truths.

The primary optimization emphasis should remain exact ENTRY and COLLISION localization.

---

# 44. Representative Tensor Shapes for T = 500

Assume:

- \(T=500\);
- DINO resolution 384;
- DINO-B dimension 768;
- \(K=12\) objects;
- model dimension 384;
- V-JEPA input 32;
- V-JEPA-L dimension 1024.

## DINO raw dense output

Per extraction chunk:

```text
[B_img, 24, 24, 768]
```

Full logical video:

```text
[500, 24, 24, 768]
```

but never retain all of it in GPU memory simultaneously.

## DINO scene embeddings

```text
[500, 768]
```

## DINO ROI appearance embeddings

```text
[500, 12, 768]
```

## Projected appearance

```text
[500, 12, 256]
```

## Geometry raw

```text
[500, 12, 9]
```

## Geometry embedded

```text
[500, 12, 128]
```

## Object tokens

```text
[500, 12, 384]
```

## Scene token

```text
[500, 384]
```

## Spatial Transformer input

Logically:

```text
[500, 13, 384]
```

The 500 frames can be processed as a batch dimension because spatial interaction is frame-local.

## Local native-frame sequence

```text
L: [500, 384]
```

## V-JEPA input

```text
[1, 3, 32, 384, 384]
```

or the repository's expected layout.

## V-JEPA dense output

```text
[1, 16, 24, 24, 1024]
```

## Spatially pooled V-JEPA

```text
[1, 16, 1024]
```

## Projected global sequence

```text
G: [1, 16, 384]
```

## Cross-attention query

```text
Q: [1, 500, 384]
```

## Cross-attention key/value

```text
K,V: [1, 16, 384]
```

## Retrieved global context

```text
C: [1, 500, 384]
```

## Fused sequence

```text
F: [1, 500, 384]
```

## Final temporal sequence

```text
H: [1, 500, 384]
```

## Event logits

```text
entry_logits:     [1, 500]
collision_logits: [1, 500]
```

## State logits

```text
entry_state_logits:     [1, 500]
collision_state_logits: [1, 500]
```

---

# 45. Training Pipeline

A complete training iteration should conceptually perform the following.

## Offline preparation

```text
video
 │
 ├─ decode native frames
 │
 ├─ RF-DETR inference
 │
 ├─ Depth Anything inference
 │
 ├─ full-video tracking / trajectory construction
 │
 └─ frozen DINOv3-B extraction
       │
       ├─ scene feature
       └─ vehicle ROI features
            │
            ▼
          cache
```

## Online training

```text
cached DINO scene/ROI features
             +
cached geometry / tracks
             │
             ▼
trainable scene/object projections
             │
             ▼
per-frame spatial Transformer
             │
             ▼
local native sequence L
             │
       local temporal refinement
             │
             ├────────────────────────────────────┐
             │                                    │
             │                         sample 32 RGB frames
             │                                    │
             │                        consistent augmentation
             │                                    │
             │                           V-JEPA 2.1-L + LoRA
             │                                    │
             │                          16 global tubelet tokens
             │                                    │
             └────────── gated cross-attention ◄──┘
                              │
                              ▼
                          fused F
                              │
                    2–4 temporal blocks
                              │
                              ▼
                              H
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 ENTRY              COLLISION
                distribution        distribution
                    │                   │
                    ├──── states ───────┤
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    event-conditioned pooling
                        ┌─────┴─────┐
                        ▼           ▼
                     side        evasion
```

Backpropagation updates:

- trainable DINO scene/object projection layers;
- geometry embedding;
- spatial interaction Transformer;
- local temporal layers;
- V-JEPA LoRA;
- V-JEPA projection;
- fusion module;
- temporal refinement;
- localization heads;
- state heads;
- attribute heads.

Backpropagation does **not** update:

- DINOv3-B;
- RF-DETR;
- Depth Anything.

---

# 46. Inference Pipeline

Inference should avoid unnecessary stochasticity.

1. load cached DINO/geometry features or compute them once;
2. select deterministic 32 V-JEPA frames;
3. run V-JEPA + learned LoRA;
4. construct global tokens and timestamps;
5. construct local native-frame tokens;
6. perform gated cross-attention;
7. run temporal refinement;
8. compute ENTRY/COLLISION distributions;
9. perform constrained joint decode \(E\le C\);
10. predict `entry_side`;
11. predict `evasion_space`.

Return native frame IDs directly.

No conversion from coarse bins is required.

---

# 47. Variable-Length Videos and Masks

Batching videos of different lengths requires padding.

For batch item \(b\), define valid length:

\[
T_b.
\]

Pad local sequences to:

\[
T_{max}^{batch}.
\]

Use:

```text
local_valid_mask: [B, T_max]
```

Mask padded frames in:

- local temporal self-attention;
- event distributions;
- state losses;
- event pooling;
- interval pooling.

Before event softmax, assign padded logits:

\[
-\infty.
\]

The V-JEPA branch always receives the configured sparse sample count, for example 32, drawn from valid frames only.

---

# 48. Horizontal Flip Augmentation

If horizontal flipping is used anywhere that affects semantic labels, remember:

\[
LEFT \leftrightarrow RIGHT.
\]

Therefore:

- swap `entry_side`;
- horizontally transform object coordinates;
- transform detector boxes consistently.

If DINO is cached from unflipped frames, arbitrary online horizontal flipping of its appearance features is not directly possible unless:

1. a mirrored DINO cache is also precomputed; or
2. horizontal flip is disabled for the cached DINO branch.

A practical approach is to precompute original and mirrored DINO features if horizontal-flip augmentation proves useful.

---

# 49. Compute Characteristics

## Frozen DINO

DINO extraction is an offline preprocessing cost.

For roughly 500 frames per video, frozen DINOv3-B inference is practical on a 4090-class GPU when batched.

Because DINO is cached, this cost is not repeated every epoch.

## V-JEPA

Only 32 sparse frames are passed through the trainable V-JEPA global branch.

This is far cheaper than passing all 500 native frames through V-JEPA.

## Fusion

For:

\[
T=500,\quad N_g=16
\]

cross-attention complexity is tiny:

\[
O(500\times16).
\]

## Temporal network

A 500×384 sequence is small enough that 2–4 lightweight temporal blocks are feasible.

---

# 50. Why This Architecture Is Preferable to the Current Coarse-to-Fine Cascade

## Current cascade

```text
full video
   ↓
coarse 64-bin prediction
   ↓
hard region selection
   ↓
fine localizer
```

Failure:

```text
bad coarse prediction
    ↓
correct frame excluded
    ↓
fine model cannot recover
```

## New architecture

```text
every native frame
       +
global sparse context
       ↓
full-video event distribution
```

Every native frame remains a candidate throughout inference.

The model can be wrong, but it cannot fail merely because a router removed the correct frame from consideration.

---

# 51. Expected Strengths

## 51.1 No error accumulation from hard routing

There is no discrete coarse prediction that determines what the precise localizer can see.

## 51.2 Native-frame DINO precision

Every decoded frame gets its own spatial representation.

There is no tubelet-size-2 bottleneck in the local branch.

## 51.3 V-JEPA's role is well matched

V-JEPA does not have to solve ±1-frame localization.

It supplies the type of information it is better suited for:

- video context;
- temporal semantics;
- accident evolution.

## 51.4 Existing geometry investment is preserved

The RF-DETR/depth/object representation remains useful and becomes available at every native frame.

## 51.5 Cheap iterative experimentation

Once DINO features are cached, most architecture experiments do not need to rerun DINO.

---

# 52. Main Risks

## 52.1 Frozen DINO may be overly invariant temporally

Neighboring DINO representations may be highly similar.

Mitigation:

- explicit differences;
- local temporal Conv1D;
- discriminability-oriented temporal blocks;
- before/after state supervision.

## 52.2 V-JEPA may be ignored

There are only 16 V-JEPA tokens and the DINO branch is strong.

Mitigation:

- inspect cross-attention entropy;
- inspect gate values;
- use auxiliary global-context dropout experiments;
- compare against DINO-only baseline.

Do not artificially force a large V-JEPA contribution unless validation proves it helps.

## 52.3 V-JEPA may overwhelm local precision

If global context is added without control, sparse semantics may blur exact frame distinctions.

Mitigation:

\[
F=L+\gamma C
\]

with conservative gate initialization.

## 52.4 Overfitting

The labeled dataset is small.

Mitigation:

- frozen DINO;
- LoRA rather than full V-JEPA fine-tuning initially;
- small temporal/fusion modules;
- strong regularization;
- cross-validation or stable source-aware splits;
- carefully limited model depth.

## 52.5 Source/domain bias

Nexar and AI-Hub may differ in:

- camera tone;
- FPS;
- compression;
- framing;
- geography;
- accident composition.

Metrics should be broken down by source.

---

# 53. Required Ablation Plan

Do not change all components simultaneously without measuring their value.

## A0 — local baseline

```text
DINO + geometry
→ spatial interaction
→ direct event heads
```

Purpose:

Measure how much can already be solved using frozen native-frame local features.

## A1 — local temporal refinement

```text
DINO + geometry
→ spatial interaction
→ 2–4 temporal blocks
→ event heads
```

Difference from A0 isolates native-frame temporal modeling.

## A2 — add V-JEPA fusion

```text
A1
+ V-JEPA global tokens
+ gated cross-attention
```

This is the main proposed architecture.

## A3 — ungated residual

Compare:

\[
F=L+C
\]

against:

\[
F=L+\gamma C.
\]

This determines whether the gate actually protects/localizes precision.

## A4 — no local identity

Test:

\[
F=C
\]

only as a diagnostic.

It is expected to be worse because cross-attention output is a mixture of sparse V-JEPA values.

## A5 — V-JEPA 32 vs 64 input frames

Compare:

```text
32 input frames → 16 global tokens
```

against:

```text
64 input frames → 32 global tokens
```

## A6 — difference features

Compare with/without:

\[
\Delta_t
\]

and second difference.

## A7 — temporal network depth

Compare:

- direct heads;
- 2 blocks;
- 4 blocks;
- encoder-decoder.

## A8 — V-JEPA LoRA vs frozen

Measure whether V-JEPA adaptation provides enough value to justify online training.

## A9 — geometry ablation

Compare:

- DINO only;
- DINO + boxes;
- DINO + full 9-D geometry/depth.

## A10 — object ROI ablation

Compare:

- global DINO frame token only;
- global + tracked vehicle ROI tokens.

---

# 54. Recommended Evaluation Metrics

Do not rely only on the final competition aggregate.

For ENTRY and COLLISION separately measure:

## Exact accuracy

\[
Acc@0
\]

prediction exactly equals ground truth.

## Tolerance accuracy

\[
Acc@\pm1
\]

\[
Acc@\pm2
\]

\[
Acc@\pm3.
\]

## Mean absolute frame error

\[
MAE_{frames}
=
\frac1N
\sum_i
|\hat t_i-t_i|.
\]

## Time-normalized error

\[
MAE_{norm}
=
\frac1N
\sum_i
\left|
\frac{\hat t_i-t_i}{T_i-1}
\right|.
\]

## Seconds error

Use each sample's native FPS.

## Distribution calibration

Inspect:

- max probability;
- entropy;
- probability mass within ±1/±2 frames of ground truth.

For attributes measure:

- side accuracy / F1;
- evasion accuracy / F1.

Break all metrics down by:

- source;
- FPS;
- duration;
- intersection/road type where available.

---

# 55. Debugging Visualizations

For every validation video, save a temporal plot showing:

```text
time ─────────────────────────────────────►

ENTRY probability
COLLISION probability
ENTRY state sigmoid
COLLISION state sigmoid
global cross-attention gate
GT ENTRY
GT COLLISION
pred ENTRY
pred COLLISION
```

Also inspect:

- cross-attention maps \(A_{tj}\);
- average gate \(\gamma_t\);
- object tracks near predicted events;
- DINO frame-feature cosine similarity around event boundaries.

These visualizations are especially important for determining whether the model is:

- locally confused;
- globally confused;
- overusing V-JEPA;
- ignoring V-JEPA;
- over-smoothing neighboring frames.

---

# 56. Recommended Initial Hyperparameters

These are starting values.

## Local branch

```yaml
dino:
  model: dinov3_vitb16
  frozen: true
  lora: false
  image_size: 384

objects:
  max_tracks: 12
  dino_roi_dim: 768
  roi_projected_dim: 256
  geometry_raw_dim: 9
  geometry_dim: 128
  object_dim: 384

spatial:
  dim: 384
  depth: 2
  heads: 6
```

## Global branch

```yaml
vjepa:
  model: vjepa2_1_vitl
  image_size: 384
  num_input_frames: 32
  tubelet_size: 2
  feature_dim: 1024
  projected_dim: 384
  lora_rank: 16
  lora_alpha: 16
  lora_dropout: 0.05
```

## Fusion

```yaml
fusion:
  dim: 384
  heads: 6
  relative_time_bias: true
  gate: scalar_or_grouped
  gate_init_bias: -2.0
```

## Temporal refinement

```yaml
temporal:
  dim: 384
  num_blocks: 2
  preserve_native_resolution: true
```

Start with 2 blocks.

Move to 4 only if justified.

## Localization

```yaml
localization:
  gaussian_sigma_frames: 1.0
  position_loss_weight: 0.2
  state_loss_weight: 0.2
  order_loss_weight: 0.05
```

---

# 57. Optimizer Strategy

Because the modules differ significantly in pretraining status, use parameter groups.

Example:

## V-JEPA LoRA

Low learning rate:

\[
1e{-5}\text{ to }5e{-5}.
\]

## New fusion / spatial / temporal / heads

Higher learning rate:

\[
1e{-4}\text{ to }3e{-4}.
\]

## DINO

No optimizer parameters.

## Detector/depth

No optimizer parameters.

Use AdamW.

A reasonable starting policy:

```yaml
optimizer: AdamW
new_module_lr: 2e-4
vjepa_lora_lr: 3e-5
weight_decay: 0.05
```

Tune from validation.

---

# 58. Training Precision and Memory

Use BF16 where supported.

DINO is not part of the training graph, so its extraction memory is separate from the live training model.

For online training, major memory consumers are:

- V-JEPA activations;
- temporal/local fusion layers;
- object-token processing.

If necessary:

- gradient checkpoint V-JEPA last trainable blocks;
- reduce V-JEPA global frames from 64 to 32;
- keep local feature caches on CPU and move only batch tensors to GPU.

---

# 59. DINO Cache File Design

A practical per-video cache could contain:

```python
{
    "sample_id": ...,
    "num_frames": T,
    "fps": ...,
    "scene_features": Tensor[T, 768],
    "roi_features": Tensor[T, K, 768],
    "roi_valid": BoolTensor[T, K],
    "track_ids": LongTensor[T, K],
}
```

Geometry may be stored separately or together:

```python
"geometry": Tensor[T, K, 9]
```

Recommended storage:

- `.pt`, `.safetensors`, or chunked array format;
- BF16/FP16 for appearance features;
- FP16/FP32 for geometry depending on numerical needs.

The cache should have an explicit version string tied to:

- DINO checkpoint;
- resize/crop preprocessing;
- tracker version;
- ROIAlign coordinate logic.

Otherwise stale caches can silently produce inconsistent experiments.

---

# 60. No Separate Motion Branch

This architecture intentionally does **not** introduce a separate optical-flow or motion-estimation branch.

Temporal information is obtained from:

1. V-JEPA global video representation;
2. temporal changes in DINO/object features;
3. cached geometric trajectories;
4. temporal refinement blocks.

If explicit optical flow is ever revisited, it should be treated as a later ablation, not part of the base architecture.

---

# 61. Implementation Order

Recommended coding sequence:

## Step 1

Build the new full-video dataset loader that returns:

- variable-length native sequence metadata;
- cached DINO scene/ROI features;
- geometry/tracks;
- original RGB access for V-JEPA sampled frames;
- all four labels.

## Step 2

Implement DINO cache generation.

Verify exact alignment among:

- native frame index;
- detector box;
- DINO feature map;
- ROI feature.

## Step 3

Implement local per-frame object interaction.

Confirm:

```text
[T, 13, 384] → [T, 384]
```

## Step 4

Implement DINO-only direct event baseline.

Do this before V-JEPA fusion.

## Step 5

Add full-resolution local temporal refinement.

## Step 6

Add sparse V-JEPA branch.

Confirm:

```text
32 RGB frames
→ 16 × 24 × 24 × 1024
→ 16 × 1024
→ 16 × 384
```

## Step 7

Implement relative-time gated cross-attention.

## Step 8

Implement event-query heads.

## Step 9

Implement state auxiliary heads.

## Step 10

Implement event-conditioned side/evasion heads.

## Step 11

Implement constrained joint decoding.

## Step 12

Add all ablation switches to config rather than maintaining separate model forks.

---

# 62. Recommended Configurable Switches

Expose:

```yaml
model:
  use_geometry: true
  use_roi_appearance: true
  use_local_difference: true

  use_local_temporal_pre_fusion: true

  use_vjepa: true
  vjepa_num_frames: 32

  fusion:
    type: gated_cross_attention
    relative_time_bias: true
    preserve_local_residual: true

  post_fusion_temporal:
    type: full_resolution
    depth: 2

  event_head:
    type: query_pointer
    use_query_global_conditioning: false

  auxiliary:
    state_heads: true
    order_loss: true

  attributes:
    entry_conditioned_side: true
    interval_conditioned_evasion: true
```

This enables controlled ablations without rewriting the architecture.

---

# 63. Final Recommended Base Model

The first serious model to train should be:

```text
Frozen DINOv3-B/16 @384 on every native frame
        │
        ├─ scene DINO feature
        └─ top-12 tracked vehicle ROI features
                     +
             9-D geometry/depth
                     │
                     ▼
         2-layer spatial Transformer
                     │
                     ▼
               T × 384 local
                     │
            local temporal Conv
                     │
                     │ Q
                     ▼
        gated cross-attention
                     ▲
                     │ K,V
     V-JEPA 2.1 ViT-L + LoRA
       32 stratified RGB frames
         → 16 global tokens
                     │
                     ▼
               T × 384 fused
                     │
       2 full-resolution temporal blocks
                     │
                     ▼
               T × 384 final
             ┌───────┴───────┐
             ▼               ▼
        ENTRY query      COLLISION query
             │               │
             ▼               ▼
          p_E(1:T)        p_C(1:T)
             │               │
             └──────┬────────┘
                    ▼
          constrained E ≤ C decode
                    │
       ┌────────────┴─────────────┐
       ▼                          ▼
ENTRY-conditioned           E→C interval-conditioned
entry_side                  evasion_space
```

---

# 64. Central Hypothesis

The architecture is based on a very specific hypothesis:

> **Exact event localization and global accident understanding should not be forced into the same temporal representation scale.**

DINOv3 processes every native frame and preserves spatial/frame-level detail.

V-JEPA processes a sparse global sample and learns the accident's broader temporal semantics.

The fusion mechanism lets each native frame retrieve only the global information relevant to it, while the residual identity path guarantees that sparse global context cannot erase the exact local evidence needed for ±1-frame localization.

Formally:

\[
\boxed{
F_t
=
L_t
+
\gamma_t
P\left(
Attn(
L_t,
G,
G
)
\right)
}
\]

followed by lightweight native-resolution temporal reasoning.

ENTRY and COLLISION are then solved directly as:

\[
\boxed{
p_E(t), p_C(t), \qquad t=0,\ldots,T-1
}
\]

rather than through a coarse route followed by a fine search.

This is the defining difference from the previous Stage 2 model.

---

# 65. What Should Not Be Added Initially

Avoid adding all of the following until the base ablations are complete:

- full DINO fine-tuning;
- DINO LoRA;
- a separate optical-flow branch;
- two-phase V-JEPA;
- a deep temporal U-Net;
- a large full-video Transformer over raw DINO patch tokens;
- DETR-style multiple temporal proposals;
- a second coarse routing model;
- large ensembles.

The architecture already contains several strong pretrained components. The first goal is to measure whether the **local-native + sparse-global decomposition** solves the Stage 2 bottleneck.

---

# 66. Decision Criteria After the First Experiments

## If DINO-only is already very strong

Keep the global branch small.

V-JEPA may add only modest gains.

## If DINO has low MAE but weak ambiguous ENTRY cases

The V-JEPA fusion is likely valuable.

Inspect gates and cross-attention.

## If V-JEPA improves overall metrics but worsens ±1-frame accuracy

Reduce global gate strength or use stronger local residual protection.

## If predictions are temporally broad

Improve local temporal discriminability:

- difference features;
- SGP-style blocks;
- state auxiliary loss;
- smaller Gaussian sigma.

## If predictions choose the wrong accident phase

Increase global context:

- 64 V-JEPA frames;
- stronger query conditioning;
- slightly deeper global fusion.

## If the model overfits rapidly

Simplify:

- fewer temporal blocks;
- scalar rather than channel gate;
- frozen V-JEPA baseline;
- lower projection/head dimensions;
- stronger feature dropout.

---

# 67. Summary

The proposed Stage 2 v2 model is a **single-stage dual-branch precise event spotter**.

It replaces:

```text
coarse localization
→ hard window routing
→ fine localization
```

with:

```text
native-frame local representation
+
sparse full-video global representation
→ controlled fusion
→ full-video exact-frame prediction
```

The local stream is:

\[
\text{Frozen DINOv3-B}
+
\text{object ROI appearance}
+
\text{geometry/depth}
\]

and produces:

\[
L\in\mathbb{R}^{T\times384}.
\]

The global stream is:

\[
\text{V-JEPA 2.1 ViT-L + LoRA}
\]

on 32 stratified frames and produces:

\[
G\in\mathbb{R}^{16\times384}.
\]

Fusion is:

\[
C=Attn(L,G,G)
\]

followed by:

\[
F=L+\gamma\odot P(C).
\]

After 2–4 lightweight native-resolution temporal blocks, two event-query heads predict:

\[
p_E(1:T)
\]

and:

\[
p_C(1:T).
\]

ENTRY and COLLISION are jointly decoded under:

\[
ENTRY\le COLLISION.
\]

`entry_side` is predicted from ENTRY-conditioned features.

`evasion_space` is predicted from the ENTRY→COLLISION interaction interval plus global context.

This architecture should be treated as the main Stage 2 v2 candidate, with the DINO-only and no-temporal-refinement variants serving as essential baselines.
