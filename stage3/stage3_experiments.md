First inspect the repository carefully, especially the current Stage 3 implementation, configs, experiment reports, preprocessing, decoder, loss functions, evaluation code, and latest handoff/experiment results\. Do not blindly trust this prompt if it conflicts with the repository\. Preserve the existing training/evaluation pipeline wherever possible\.

## Goal

Test whether representations pretrained specifically for end\-to\-end driving transfer better to DACON Stage 3 than the current handcrafted motion pipeline based on SEA\-RAFT \+ geometry/physics \+ TCN/SSM\.

Use the existing current\-best Stage 3 model as the control\.

For fast screening, use:

- BATON only
- same current train/validation split
- 20 epochs
- same seeds/settings wherever possible
- same competition evaluation code
- current best decoding thresholds:
  - acceleration threshold: ±0\.4 m/s²
  - steering threshold: ±7°
- preserve the official STOPPED handling:
  - steering F1 excludes GT STOPPED frames
  - competition score = `0.7 * acceleration_macro_f1 + 0.3 * steering_macro_f1`

Do not make unrelated changes to the decoder, dataset, augmentation, or metric during these representation experiments\.

---

# Experiment 0 — Reproduce control

Run the current best Stage 3 architecture for 20 epochs under exactly the same BATON configuration that will be used for the new models\.

Record at minimum:

- competition score
- acceleration Macro\-F1
- steering Macro\-F1
- per\-class F1
- continuous acceleration MAE
- steering\-angle MAE
- speed MAE
- parameter count
- peak VRAM
- training/inference speed

This is the reference for every following comparison\.

---

# Experiment 1 — Pretrained TCP\-ResNet34 → Stage 3

Investigate the official/public implementation and pretrained weights for **TCP: Trajectory\-guided Control Prediction**\.

Do not simply copy its CARLA control output\.

We want to test whether its pretrained **driving\-specific visual representation** transfers to Stage 3\.

## 1A\. TCP frozen encoder

Use the pretrained TCP ResNet\-34 visual encoder\.

Input:
`front RGB video @ Stage 3 10 Hz`

Extract per\-frame or short\-context TCP visual features\.

Do NOT give the network DACON\-unavailable inputs such as:

- ground\-truth vehicle speed
- route/navigation command
- GPS
- privileged CARLA state

If TCP normally requires those inputs, modify the architecture so that the Stage 3 representation depends only on information available at inference\.

Architecture:

```text
RGB @ 10 Hz
   ↓
pretrained TCP ResNet34
   ↓
per-frame driving feature
   ↓
small temporal model
   ↓
Stage 3 heads:
  - aEgo
  - speed
  - STOP probability
  - steering angle
```

Prefer reusing the current Stage 3 TCN\+SSM temporal block and current regression heads initially so that this experiment mainly tests the **representation**, not many changes at once\.

Keep the TCP encoder frozen in the first experiment\.

Train for 20 epochs\.

## 1B\. TCP partial fine\-tuning

Starting from 1A, unfreeze only a small part of TCP:

- preferably final ResNet block / final stage
- use a significantly smaller LR for pretrained parameters than the Stage 3 temporal/head parameters

Do not immediately full\-finetune the complete encoder\.

Train for 20 epochs\.

Compare 1A vs 1B vs current V3\.

## Optional 1C — TCP trajectory latent

If the pretrained TCP checkpoint exposes a useful trajectory\-prediction representation, test whether adding its trajectory latent helps:

```text
TCP visual feature
      +
TCP trajectory latent
      ↓
Stage3 temporal model
```

Do not treat CARLA throttle/brake values as CAN acceleration labels\.

The purpose is to reuse the driving representation, not CARLA actuator semantics\.

---

# Experiment 2 — Small driving student distilled from openpilot

The current openpilot driving model is too large to use directly as the primary Stage 3 inference model, so use it **offline as a teacher only**\.

Investigate the currently available openpilot 0\.11\.x driving model and its inference interface\.

The teacher should process BATON front\-camera frames and provide whatever valid outputs can be extracted, such as:

- desired longitudinal acceleration
- desired curvature
- future velocity trajectory
- future acceleration trajectory
- future orientation / yaw\-related trajectory
- other compact driving\-plan outputs that are clearly documented

Do not assume an output exists without checking the actual model/interface\.

Cache teacher outputs offline so openpilot is never needed during student training or DACON inference\.

## Student architecture

Use a lightweight model suitable for submission\.

Preferred initial student:

```text
RGB @ 10 Hz
   ↓
pretrained TCP ResNet34
   ↓
visual driving feature
   ↓
TCN+SSM
   ↓
Stage3 heads
```

The student must work from RGB alone at inference\.

### Distillation objectives

Use openpilot as privileged supervision\.

Do NOT simply teach the student to copy the final discrete Stage 3 labels\.

Instead distill richer driving dynamics\.

Potential teacher signals:

### Longitudinal

Student predicts:

- openpilot desired acceleration
- openpilot future acceleration sequence
- openpilot future velocity sequence

### Lateral

Student predicts:

- desired curvature
- future curvature / orientation\-related quantities if available

These teacher heads should be auxiliary during training\.

Primary supervised targets remain the actual BATON CAN targets:

- direct longitudinal acceleration `aEgo`
- steering\-wheel angle
- speed
- STOP target

Conceptually:

```text
                     Openpilot 0.11.x
BATON RGB ──────────────→ teacher
 │                          │
 │                   cached dynamics
 │                          │
 ↓                          ↓
TCP-Res34 student ─→ representation
        │
       TCN+SSM
        │
        ├── CAN aEgo
        ├── CAN steering angle
        ├── speed
        ├── STOP
        │
        └── teacher distillation heads
             ├── desired acceleration
             ├── curvature
             └── future dynamics
```

Loss:

`L_total = L_current_stage3 + λ_distill * L_teacher`

Keep the existing Stage 3 loss as the base\.

Test at least:

- `λ_distill = 0` — TCP baseline
- small distillation weight
- moderate distillation weight

Do not perform a huge hyperparameter search\.

Use the 20\-epoch screen to determine whether the signal is useful\.

---

# Important ablations

The goal is to understand **where any improvement comes from**\.

At minimum compare:

|ID|Model                                           |
|--|------------------------------------------------|
|C0|Current V3 20-epoch control                     |
|T1|frozen pretrained TCP-Res34 + current TCN/SSM   |
|T2|partially fine-tuned TCP-Res34 + current TCN/SSM|
|D1|T1 + openpilot distillation                     |
|D2|T2 + openpilot distillation                     |

If compute permits, add:

`D3 = lightweight non-TCP student + openpilot distillation`

This determines whether improvement comes mainly from TCP pretraining or from the openpilot teacher\.

---

# Critical evaluation

Do not judge experiments only by total validation score\.

Report:

### Acceleration

- Macro\-F1
- ACCELERATING F1
- DECELERATING F1
- CONSTANT F1
- STOPPED F1
- confusion matrix
- especially:
  - ACCEL → CONSTANT
  - DECEL → CONSTANT
- acceleration MAE

### Steering

- Macro\-F1
- LEFT / RIGHT / STRAIGHT F1
- steering\-angle MAE

### Physical consistency

- speed MAE
- acceleration\-vs\-speed\-derivative consistency if already supported

### Efficiency

- trainable parameters
- total parameters
- peak VRAM
- training time
- inference ms/frame
- model/checkpoint size

A candidate that improves validation by 0\.003 but doubles inference cost is not automatically useful\.

---

# Extra robustness test

Because BATON validation has known domain limitations, run simple robustness comparisons on the best control and best TCP/distilled candidate if feasible:

- brightness/contrast shift
- mild blur/compression
- spatial resize/resolution change
- temporal frame dropping / stride change while preserving the 10 Hz model interface appropriately

Do not tune specifically to these corruptions\.

We want to know whether driving\-specific pretraining gives a more transferable representation than the current SEA\-RAFT physics pipeline\.

---

# Decision criteria

The key research questions are:

1. Does pretrained TCP representation outperform the current SEA\-RAFT/physics representation under the same 20\-epoch training budget?
2. Does partial TCP fine\-tuning help, or does it overfit BATON?
3. Does openpilot teacher distillation provide information that TCP pretraining alone does not?
4. Which component improves ACCEL/DECEL vs CONSTANT discrimination?
5. Does the new model preserve or improve steering performance?
6. Does it reduce the current V3 speed/physical\-calibration weakness?
7. Is the improvement large enough to justify inference cost?

Do not combine every idea immediately\.

First identify whether:

- TCP itself helps;
- openpilot distillation itself helps;
- both are complementary\.

Only if the 20\-epoch screen is promising should a later experiment train the winning configuration for the full \~100\-epoch schedule\.

---

# Deliverables

Create a report such as:

`reports/stage3_tcp_openpilot_transfer.md`

Include:

1. exact pretrained checkpoints and upstream repositories used
2. licenses
3. architecture diagrams
4. preprocessing differences
5. any modifications required to make TCP inference RGB\-only
6. exact openpilot outputs used for distillation
7. experiment configs and commands
8. parameter count / VRAM / runtime
9. complete metrics table
10. confusion matrices
11. comparison against current V3
12. clear conclusions about which representation transferred and why
13. recommended next experiment

Keep all new experiments isolated from the production Stage 3 implementation until their value is demonstrated\.

Do not overwrite existing checkpoints or experiment results\.