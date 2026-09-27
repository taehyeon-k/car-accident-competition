Stage 2 v12 (robust, mixed backbones): v11's three XSbU families (E4 / E2 / XN4 on geometry-adapted DINOv3 ViT-S features) plus one
XSbUB_E4 family trained with the same robust recipe (equal 1/1-1/2-1/3 stride mix, 629 dedup MM-AU/CCD extras, unlabelled
cross-frame-rate consistency 0.3) on FROZEN ORIGINAL DINOv3 ViT-B features (backbone_vitb.pth = official dinov3_vitb16 weights).
4 seeds per family (16 members), full-data refit; stop epochs E4 10, E2 8, XN4 15, ViT-B E4 9. The runtime encodes the sampled frames
with both backbones and gives each member the features it was trained on (config member field "backbone").
Selection (reports/stage2_robustness_campaign.md, duplicate-clean CV, 3 seeds/family, plain decoding):
  native / 1/2 / 1/3 / 1/4 frame rate = 0.763 / 0.766 / 0.745 / 0.745 (v11 0.758 / 0.758 / 0.736 / 0.735; v10 0.768 / 0.762 / 0.729 / 0.736).
  Backbone diversity improved every frame rate; the ViT-B family alone is the best single family at 1/3 rate (0.740).
Cost: +343 MB (ViT-B weights) and a second backbone pass per sampled frame at inference.
Stage 3: V3 unchanged, steering 7.0 deg / acceleration +-0.4 (as v10/v11). Stage 1 unchanged.
