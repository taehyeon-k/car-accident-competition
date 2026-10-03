"""Build an offline submission containing the retained code and verified weights."""
import argparse
import json
from pathlib import Path
import zipfile
from fetch_checkpoints import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/submit_best_recorded.zip')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    for item in manifest['checkpoints']:
        path = ROOT / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            raise ValueError(f"Missing or mismatched checkpoint: {item['path']}; run scripts/fetch_checkpoints.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / 'submission-requirements.txt', 'requirements.txt')
        for path in [ROOT / 'inference.py', *sorted((ROOT / 'code').rglob('*')), *sorted((ROOT / 'model').rglob('*'))]:
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{args.output} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
