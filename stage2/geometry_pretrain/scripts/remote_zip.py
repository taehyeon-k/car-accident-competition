"""Selective extraction from a remote zip over HTTP range requests.

Used for TuSimple: the only complete public mirror is a 23 GB Kaggle bundle,
while this project needs a few frames per clip. Reading the central directory
and fetching selected members avoids downloading the whole archive.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()
        head = self.session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                data = r.content
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def resolve_kaggle(dataset: str) -> str:
    r = requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{dataset}", allow_redirects=False, timeout=60)
    return r.headers["location"]


def list_members(url: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)) as zf:
        return zf.infolist()


def extract(url: str, members: list[str], output: Path, workers: int = 16) -> int:
    local = threading.local()

    def get_zip():
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 18))
        return local.zf

    def one(name):
        target = output / name
        if target.is_file() and target.stat().st_size > 0:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        data = get_zip().read(name)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(target)
        return len(data)

    with ThreadPoolExecutor(workers) as pool:
        total = 0
        for i, n in enumerate(pool.map(one, members)):
            total += n
            if i % 2000 == 0:
                print(f"{i}/{len(members)} files, {total / 1e9:.2f} GB", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", default="manideep1108/tusimple")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--members-file")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    url = resolve_kaggle(args.kaggle)
    if args.list:
        for info in list_members(url):
            print(info.filename, info.file_size, info.compress_size)
        return
    members = [line.strip() for line in open(args.members_file) if line.strip()]
    total = extract(url, members, Path(args.output), args.workers)
    print(f"extracted {len(members)} members, {total / 1e9:.2f} GB new")


if __name__ == "__main__":
    main()
