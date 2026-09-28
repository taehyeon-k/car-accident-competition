"""Night campaign evaluation daemon: evaluates queued arms with the ENTRY suite as soon as their checkpoints exist.

Queue file stage2/night/eval_queue.txt, one line per job: RUN|SEEDS|TAG[|EXTRA suite args]  (RUN may join families with '+').
A job is ready when every family has fold0-4 checkpoints for every listed seed. Results are appended to stage2/night/night.log.
Done jobs are recorded in stage2/night/eval_done.txt. OBJ_CACHE defaults to cache_objlane_full (members with their own caches route
themselves). Usage: nohup python -m stage2.night.eval_daemon &
"""
from __future__ import annotations

import subprocess, time
from pathlib import Path

R = Path("/workspace/car-accident"); Q = R / "stage2/night/eval_queue.txt"; DONE = R / "stage2/night/eval_done.txt"; LOG = R / "stage2/night/night.log"
ROOTS = [R / "stage2/generalization/results", R / "stage2/aux_signal_experiments/results"]


def ready(run, seeds):
    for fam in run.split("+"):
        root = next((r for r in ROOTS if (r / fam).is_dir()), None)
        if root is None or any(not (root / fam / "cv" / f"fold{f}_seed{s}" / "checkpoint.pt").exists() for f in range(5) for s in seeds): return False
    return True


def main():
    while True:
        done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
        jobs = [l.strip() for l in (Q.read_text().splitlines() if Q.exists() else []) if l.strip() and not l.startswith("#")]
        todo = [j for j in jobs if j not in done]
        for j in todo:
            run, seeds, tag, *extra = j.split("|")
            if not ready(run, seeds): continue
            cmd = (f"cd {R} && OBJ_CACHE=stage2/objtrack/cache_objlane_full /venv/main/bin/python -m stage2.generalization.entry_suite '{run}' "
                   f"--seeds {' '.join(seeds)} --tag {tag} {' '.join(extra)} 2>&1 | grep -v Warn | grep -v '^$' >> {LOG}")
            subprocess.run(["bash", "-c", cmd]); DONE.open("a").write(j + "\n")
            break
        else:
            time.sleep(30)


if __name__ == "__main__": main()
