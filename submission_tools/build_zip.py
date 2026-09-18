"""Create the DACON ZIP once all final checkpoints have been placed."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, default=Path(__file__).resolve().parents[1]/'submission')
    parser.add_argument('--output', type=Path, default=Path('submit.zip'))
    args = parser.parse_args()
    root = args.submission.resolve()
    required = ['inference.py', 'requirements.txt', 'model/stage1/best.pt',
                'model/stage2/runtime.py', 'model/stage2/backbone.pth', 'model/stage2/probe.pt',
                'model/stage2/config.json', 'model/stage2/sampling.py',
                'model/stage3/best.pt', 'model/stage3/pretrained/sea_raft/model.safetensors']
    missing = [p for p in required if not (root/p).is_file()]
    if missing: raise SystemExit('Missing required assets: ' + ', '.join(missing))
    files = [root/'inference.py', root/'requirements.txt']
    files += [p for p in (root/'model').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}]
    if sum(p.stat().st_size for p in files) > 32_000_000_000: raise SystemExit('Exceeds unpacked 32 GB limit')
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files: z.write(p, p.relative_to(root))
    if args.output.stat().st_size > 10_000_000_000: raise SystemExit('ZIP exceeds 10 GB limit')
    print(f'{args.output.resolve()} ({args.output.stat().st_size/2**20:.1f} MiB)')


if __name__ == '__main__': main()
