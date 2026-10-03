"""Restore only the retained checkpoints from their checksum-pinned R2 archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded v8 archive')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    pending = [item for item in manifest['checkpoints']
               if not (ROOT / item['path']).is_file()
               or digest(ROOT / item['path']) != item['sha256']]
    if not pending:
        print('All 16 retained checkpoints verified.')
        return
    with tempfile.TemporaryDirectory(prefix='car-accident-checkpoints-') as folder:
        archive = args.archive
        if archive is None:
            archive = Path(folder) / 'v8.zip'
            print('Downloading the recorded v8 archive (260 MB)...', flush=True)
            urllib.request.urlretrieve(manifest['archive']['url'], archive)
        if archive.stat().st_size != manifest['archive']['size_bytes'] or digest(archive) != manifest['archive']['sha256']:
            raise ValueError('Archive checksum or size mismatch')
        with zipfile.ZipFile(archive) as source:
            for item in pending:
                destination = ROOT / item['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = source.read(item['archive_member'])
                if len(data) != item['size_bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError(f"Checkpoint checksum mismatch: {item['path']}")
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                temporary.replace(destination)
                print(f"Restored {item['path']}")
    print('All retained checkpoints verified.')


if __name__ == '__main__':
    main()
