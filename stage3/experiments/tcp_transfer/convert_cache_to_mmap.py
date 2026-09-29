"""Convert the TCP NPZ feature cache to crop-readable NPY sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--delete-npz", action="store_true")
    args = parser.parse_args()
    sources = sorted(args.cache_dir.glob("*.npz"))
    if not sources:
        raise SystemExit("No NPZ cache files found")
    for index, source in enumerate(sources, 1):
        with np.load(source) as archive:
            for key in ("tokens", "tokens_flip"):
                value = archive[key]
                target = args.cache_dir / f"{source.stem}.{key}.npy"
                temp = target.with_suffix(".npy.tmp")
                with temp.open("wb") as handle:
                    np.save(handle, value, allow_pickle=False)
                mapped = np.load(temp, mmap_mode="r")
                if mapped.shape != value.shape or mapped.dtype != value.dtype or not np.array_equal(mapped, value):
                    temp.unlink(missing_ok=True)
                    raise RuntimeError(f"Conversion verification failed: {source} {key}")
                temp.replace(target)
        if args.delete_npz:
            source.unlink()
        if index == 1 or index % 100 == 0 or index == len(sources):
            print(json.dumps({"converted": index, "total": len(sources)}), flush=True)
    marker = args.cache_dir / "mmap_complete.json"
    marker.write_text(json.dumps({"clips": len(sources), "verified": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
