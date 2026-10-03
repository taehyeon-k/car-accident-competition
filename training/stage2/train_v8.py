"""Launch a retained v8 family using its actual saved full-data refit settings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['E4', 'E2', 'XN4'], required=True)
    parser.add_argument('--seed', choices=range(4), type=int, default=0)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--dense-cache', required=True)
    parser.add_argument('--residual-cache', required=True)
    parser.add_argument('--stride-cache', required=True)
    parser.add_argument('--nexar-cache', help='XN4 extras: features, motion and labels_all.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from stage2.long_context_v2_experiments import common as common
    from stage2.aux_signal_experiments import train as trainer
    cfg = json.loads((ROOT / f'training/stage2/configs/{args.family}_sa_full_seed{args.seed}.json').read_text())
    common.DENSE = Path(args.dense_cache).resolve()
    trainer.RESIDUAL = Path(args.residual_cache).resolve()
    trainer.STRIDE = Path(args.stride_cache).resolve()
    if args.family == 'XN4':
        if not args.nexar_cache: parser.error('XN4 requires --nexar-cache with the original preprocessed expansion')
        from stage2.aux_signal_experiments import nexar_labels
        nexar_labels.OUT = Path(args.nexar_cache).resolve()
    argv = ['train', '--run-id', cfg['run_id'], '--seed', str(cfg['seed']), '--base-loss', cfg['base_loss'],
            '--motion', 'both', '--stride-aug', cfg['stride_aug'], '--epochs', str(cfg['epochs']),
            '--stop-epoch', str(cfg['stop_epoch']), '--batch-size', str(cfg['batch_size']),
            '--train-split', str(Path(args.train_manifest).resolve()),
            '--val-split', str(Path(args.val_manifest).resolve()), '--output', args.output]
    if args.family == 'E2': argv += ['--boundary', 'bnd2', '--w-bnd', str(cfg['w_bnd'])]
    if args.family == 'XN4': argv += ['--extra-nexar', '--extra-labels', 'all', '--extra-entry-w', str(cfg['extra_entry_w'])]
    sys.argv = argv
    trainer.main()


if __name__ == '__main__': main()
