"""Reconstructed Stage 1 trainer; not the missing original competition trainer.

Uses the exact submitted architecture with a documented preprocessed tensor API.
See docs/training.md for recovered settings and limits of this reconstruction.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('stage1_runtime', ROOT / 'model/stage1/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class PreprocessedDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty manifest')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['tensor_path'])
        if not path.is_absolute():
            path = self.manifest.parent / path
        value = torch.load(path, map_location='cpu', weights_only=True)
        label = int(row['label'])
        if label not in (0, 1):
            raise ValueError('label must be 0=ORIGINAL or 1=RERECORDED')
        return {name: value[name].float() for name in ('patches', 'global_frames', 'burst_profiles')}, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--config', default=ROOT / 'training/stage1/recovered_config.json')
    parser.add_argument('--output', default='runs/stage1/reconstructed_g1')
    parser.add_argument('--init', choices=['imagenet', 'random', 'checkpoint'], default='imagenet')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = runtime.Stage1RebuildModel(backbone=cfg['backbone'], pretrained=args.init == 'imagenet',
                                       dropout=cfg['dropout'], modality_dropout=cfg['modality_dropout']).to(device)
    if args.init == 'checkpoint':
        state = torch.load(ROOT / 'model/stage1/best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
    train = DataLoader(PreprocessedDataset(args.train_manifest), batch_size=cfg['batch_size'], shuffle=True,
                       num_workers=cfg['num_workers'])
    val = DataLoader(PreprocessedDataset(args.val_manifest), batch_size=1, num_workers=cfg['num_workers'])
    backbone_params = list(model.rgb_backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['backbone_learning_rate']},
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': cfg['head_learning_rate']},
    ], weight_decay=cfg['weight_decay'])
    # Reconstructed scheduler: warmup + cosine. Original scheduler state is absent.
    def schedule(epoch):
        warmup = cfg['warmup_epochs']
        if epoch < warmup: return (epoch + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, cfg['epochs'] - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best = -1.0; bad = 0; history = []
    for epoch in range(cfg['epochs']):
        model.train()
        model.rgb_backbone.requires_grad_(False)
        if epoch >= cfg['freeze_backbone_epochs']:
            # Closest recoverable mapping of partial_unfreeze_blocks to ConvNeXt.
            for block in model.rgb_backbone.encoder[-cfg['partial_unfreeze_blocks']:]:
                block.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        accumulation = cfg['gradient_accumulation_steps']
        for step, (inputs, labels) in enumerate(train):
            inputs = {k:v.to(device) for k,v in inputs.items()}
            labels = labels.to(device).float()
            pred = model(**inputs)
            loss = F.binary_cross_entropy_with_logits(pred['logit'], labels)
            loss += cfg['auxiliary_loss_weight'] * (
                F.binary_cross_entropy_with_logits(pred['sparse_logit'], labels)
                + F.binary_cross_entropy_with_logits(pred['burst_logit'], labels))
            loss += cfg['global_auxiliary_loss_weight'] * F.binary_cross_entropy_with_logits(pred['global_logit'], labels)
            # Scale the last partial accumulation window by its actual size.
            window = min(accumulation, len(train) - (step // accumulation) * accumulation)
            (loss / window).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.eval(); truth = []; predictions = []
        with torch.inference_mode():
            for inputs, labels in val:
                probability = model(**{k:v.to(device) for k,v in inputs.items()})['logit'].sigmoid()
                truth.extend(labels.tolist()); predictions.extend((probability >= 0.5).cpu().int().tolist())
        score = float(f1_score(truth, predictions, labels=[0,1], average='macro', zero_division=0))
        history.append({'epoch':epoch + 1, 'val_macro_f1':score})
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if score > best:
            best = score; bad = 0
            torch.save({'model_state_dict':model.state_dict(), 'config':cfg, 'threshold':0.5,
                        'epoch':epoch, 'val_macro_f1':score, 'trainer_provenance':'reconstructed_2026-10-03'}, output / 'best.pt')
        else: bad += 1
        print(f'Epoch {epoch + 1}: val_macro_f1={score:.6f}', flush=True)
        scheduler.step()
        if bad >= cfg['early_stopping_patience']: break


if __name__ == '__main__':
    main()
