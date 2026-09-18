"""Fit and export the report's small temporal probe on cached no-anchor features.

Run inside the project container from /workspace/car-accident. No backbone training.
"""
import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2.geometry_pretrain.downstream_probe import VideoFeatures, TemporalProbe, collate, to_dev
from stage2.geometry_pretrain.common import read_jsonl
from stage2.utils.joint_losses import joint_loss, constrained_decode
from stage2.utils.joint_metrics import JointMetricAccumulator, joint_metric_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='submission_tools/learned_stage2/probe.pt')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-manifest', default='/workspace/data/stage2/manifests/train.jsonl')
    parser.add_argument('--val-manifest', default='/workspace/data/stage2/manifests/val.jsonl')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_jsonl(args.train_manifest)
    val = read_jsonl(args.val_manifest)
    train_ids = {r['sample_id'] for r in rows}
    if train_ids & {r['sample_id'] for r in val}:
        raise ValueError('Training and validation IDs overlap')
    model = TemporalProbe(dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    loader = DataLoader(VideoFeatures(rows, 'adapted_noanchor'), batch_size=8, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    vl = DataLoader(VideoFeatures(val, 'adapted_noanchor'), batch_size=1, collate_fn=collate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, total_steps=args.epochs * len(loader), pct_start=0.1)
    for epoch in range(args.epochs):
        model.train(); loss_sum = 0.0
        for batch in loader:
            batch = to_dev(batch, device)
            out = model(batch['x'], batch['time_valid'])
            loss, _ = joint_loss(out, batch, entry_sigma_seconds=0.15, collision_sigma_seconds=0.10)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step(); loss_sum += loss.item()
        print(f'epoch {epoch+1}/{args.epochs} loss={loss_sum/len(loader):.5f}', flush=True)
    model.eval(); metrics = JointMetricAccumulator(); entry_hits = collision_hits = total = 0
    with torch.inference_mode():
        for batch in vl:
            batch = to_dev(batch, device); out = model(batch['x'], batch['time_valid'])
            metrics.update(joint_metric_packet(out, batch))
            e, c = constrained_decode(out['entry_logits'], out['collision_logits'])
            es = batch['frame_seconds'].gather(1, e[:, None])[:, 0]
            cs = batch['frame_seconds'].gather(1, c[:, None])[:, 0]
            entry_hits += int(((es-batch['entry_s']).abs() <= .300001).sum())
            collision_hits += int(((cs-batch['collision_s']).abs() <= .300001).sum()); total += len(e)
    result = metrics.compute()
    result.update(acc_entry_native=entry_hits/total, acc_collision_native=collision_hits/total)
    result['competition_score_native'] = .35*(entry_hits+collision_hits)/total + .15*(result['f1_entry_side_macro']+result['f1_evasion_space_macro'])
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.cpu().state_dict(), 'model_kwargs': {'dropout': 0.3},
                'backbone_variant': 'phase1_partial_noanchor', 'seed': args.seed, 'epochs': args.epochs,
                'train_ids': sorted(train_ids), 'validation': result}, path)
    path.with_suffix('.metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
