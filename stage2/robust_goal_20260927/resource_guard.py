"""Cooperative RAM backpressure for this campaign only."""
import time
from pathlib import Path

def resource_guard(min_gib):
    while True:
        gib=next(int(s.split()[1])/2**20 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if gib>=min_gib:return
        print(f'RESOURCE_WAIT available={gib:.2f} GiB required={min_gib:.2f}',flush=True)
        time.sleep(20)

