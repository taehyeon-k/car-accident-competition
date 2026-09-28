"""Memory-aware run queue (replaces the fixed-P xargs pools).

Launches a training run only while MemAvailable >= MIN_AVAIL_GB (keeps >= 10 GB free for other jobs on the machine plus
headroom for the new run) and fewer than MAX_RUNS study runs are alive. Never kills anything. Jobs are read from jobs.txt
on every loop (lines: run_id|output_dir|train.py args), so jobs can be appended while it runs. A job is skipped when its
checkpoint exists or a live process is already training into its output dir.
"""
from __future__ import annotations

import subprocess, time
from pathlib import Path

D = Path("/workspace/car-accident/stage2/aux_signal_experiments")
JOBS = D / "jobs.txt"
MIN_AVAIL_GB, MAX_RUNS, SETTLE_S, RESERVE_GB = 3.0, 10, 20, 2.5  # recent launches count as RESERVE_GB each for 60 s
# Memory rule (user, 2026-09-28): 15 GB of the 30 GB machine are reserved for other jobs (margin included); this queue's own
# processes (training + stage2 evaluation) may use the rest: BUDGET_GB = 30 - 15 - ~1 GB system. A run launches only if
# own anonymous memory (PSS) + RESERVE_GB x recent launches + NEW_RUN_GB <= BUDGET_GB.
BUDGET_GB, NEW_RUN_GB = 14.0, 2.5


def own_rss_gb():
    tot = 0
    for p in Path("/proc").iterdir():
        try:
            cmd = (p / "cmdline").read_bytes()
            if b"python" not in cmd.split(b"\0")[0] or not (b"stage2.aux_signal_experiments.train" in cmd or b"stage2.generalization." in cmd): continue
            for line in open(p / "smaps_rollup"):  # proportional anonymous + shmem memory (file-backed cache pages are shared/reclaimable)
                if line.startswith(("Pss_Anon:", "Pss_Shmem:")): tot += int(line.split()[1])
        except Exception:
            continue
    return tot / 2**20


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
    return 0.0


def live_outputs():
    out = set()
    for p in Path("/proc").iterdir():
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except Exception:
            continue
        if b"stage2.aux_signal_experiments.train" in cmd:
            if b"--output" in cmd: out.add(cmd[cmd.index(b"--output") + 1].decode())
            else:  # default output of train.py: results/<run-id>/seed<seed>
                run = cmd[cmd.index(b"--run-id") + 1].decode(); seed = cmd[cmd.index(b"--seed") + 1].decode() if b"--seed" in cmd else "0"
                out.add(f"stage2/aux_signal_experiments/results/{run}/seed{seed}")
    return out


def ready(args):
    if "--geo" in args and len(list((D / "cache_geo").glob("*.geo.npz"))) < 349: return False
    return True


def main():
    started, attempts = {}, {}
    while True:
        jobs = [l.strip().split("|") for l in JOBS.read_text().splitlines() if l.strip() and not l.startswith("#")]
        live = live_outputs()
        todo = [(r, o, a) for r, o, a in jobs if not (Path("/workspace/car-accident") / o / "checkpoint.pt").exists() and o not in live
                and attempts.get(o, 0) < 2]
        pending = [j for j in todo if ready(j[2])]
        if not todo and not live:
            (D / "logs/batches.log").open("a").write("QUEUE_DONE\n"); return
        recent = [t for t in started.values() if time.time() - t < 60]
        if pending and len(live) < MAX_RUNS and avail_gb() >= MIN_AVAIL_GB and own_rss_gb() + RESERVE_GB * len(recent) + NEW_RUN_GB <= BUDGET_GB:
            run, out, args = pending[0]
            cmd = (f"cd /workspace/car-accident && OMP_NUM_THREADS=2 /venv/main/bin/python -m stage2.aux_signal_experiments.train "
                   f"--run-id {run} --output {out} {args} 2>&1 | grep -E 'RESULT|Error|Traceback|rror' >> {D}/logs/{run}.log")
            subprocess.Popen(["bash", "-c", cmd]); started[out] = time.time(); attempts[out] = attempts.get(out, 0) + 1
            if attempts[out] == 2: print(time.strftime("%H:%M:%S"), f"retry {out}", flush=True)
            print(time.strftime("%H:%M:%S"), f"launch {out} (avail {avail_gb():.1f} GB, own {own_rss_gb():.1f} GB, live {len(live) + 1})", flush=True)
            time.sleep(SETTLE_S)
            continue
        time.sleep(20)


if __name__ == "__main__": main()
