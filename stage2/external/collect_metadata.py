"""Download only the official metadata needed to screen MM-AU and CausalCrash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))

SOURCES = {
    "mmau": {
        "revision": "540cb1277cb70e91a7022abe852decb3ee9adb0a",
        "code_revision": "2b205a48d81f78b0c09d387d04a333aca9fc5949",
        "files": {
            "video_metadata.json": "https://raw.githubusercontent.com/jeffreychou777/LOTVS-MM-AU/2b205a48d81f78b0c09d387d04a333aca9fc5949/video_metadata.json",
            "video_metadata.csv": "https://raw.githubusercontent.com/jeffreychou777/LOTVS-MM-AU/2b205a48d81f78b0c09d387d04a333aca9fc5949/video_metadata.csv",
            "cap_text_annotations.xls": "https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/540cb1277cb70e91a7022abe852decb3ee9adb0a/cap_text_annotations.xls",
            "dada_text_annotations.xlsx": "https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/540cb1277cb70e91a7022abe852decb3ee9adb0a/dada_text_annotations.xlsx",
            "README.md": "https://raw.githubusercontent.com/jeffreychou777/LOTVS-MM-AU/2b205a48d81f78b0c09d387d04a333aca9fc5949/README.md",
        },
    },
    "causalcrash": {
        "revision": "117221526dcabe23b55333457e3b49e0654d85d6",
        "files": {
            "annotations.json": "https://huggingface.co/datasets/meet2008/CausalCrash/resolve/117221526dcabe23b55333457e3b49e0654d85d6/annotations.json",
            "videos.csv": "https://huggingface.co/datasets/meet2008/CausalCrash/resolve/117221526dcabe23b55333457e3b49e0654d85d6/videos.csv",
            "inaccessible_videos.csv": "https://huggingface.co/datasets/meet2008/CausalCrash/resolve/117221526dcabe23b55333457e3b49e0654d85d6/inaccessible_videos.csv",
            "README.md": "https://huggingface.co/datasets/meet2008/CausalCrash/resolve/117221526dcabe23b55333457e3b49e0654d85d6/README.md",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> dict:
    if destination.is_file() and not force:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "status": "EXISTING",
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "stage2-external-collector/1.0"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": "DOWNLOADED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    metadata_root = arguments.root / "metadata"
    for source, specification in SOURCES.items():
        directory = metadata_root / source
        records = []
        for name, url in specification["files"].items():
            record = download(url, directory / name, arguments.force)
            print(f"{source}: {name}: {record['status']} ({record['bytes']} bytes)")
            records.append(record)
        manifest = {
            "source": source,
            "revision": specification["revision"],
            "files": records,
        }
        if "code_revision" in specification:
            manifest["code_revision"] = specification["code_revision"]
        manifest_path = directory / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
