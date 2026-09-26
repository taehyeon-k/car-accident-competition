"""Why is NEXAR the hardest unseen source? Evaluate the NEXAR-held-out LOSO models (trained on AIHUB+CCD+MMAU only) on NEXAR clips
cropped to shorter windows that still contain ENTRY..COLLISION (crop fraction 1, 1/2, 1/4, 1/8; seeded per clip). If accuracy recovers
as context shrinks, long context (distractors) is the cause; if not, the appearance domain is. Plain direct decoding, 3-seed ensemble."""
import sys
import torch
from stage2.long_context_v2_experiments import common as C
from stage2.aux_signal_experiments.model import load as load_model
from stage2.generalization.robust_eval import item, predict

run = sys.argv[1] if len(sys.argv) > 1 else "LOSO_E4"
dev = torch.device("cuda"); R = C.REPO / "stage2/generalization/results" / run
models = []
for s in (0, 1, 2):
    ck = R / f"NEXAR_seed{s}" / "checkpoint.pt"; m = load_model(ck, dev)
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
    models.append((m, {"both": "both", "global": "global"}.get(cfg.get("motion", "none"), None)))
rows = C.rows(str(C.REPO / "stage2/generalization/loso/NEXAR_val.jsonl"))
for crop in (0.0, 0.5, 0.25, 0.125):
    b = C.metrics([predict(models, item(r, 1, crop), dev) for r in rows])
    print(f"{run} NEXAR held-out, crop {crop if crop else 1.0:5}: score {b['score']:.3f}  E {b['entry_acc']:.3f}  C {b['collision_acc']:.3f}  side {b['side_f1']:.3f}  eva {b['evasion_f1']:.3f}", flush=True)
