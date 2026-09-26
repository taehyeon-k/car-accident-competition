"""Helpers to append LOSO / CV jobs to the memory-aware queue (stage2/aux_signal_experiments/queue.py reads jobs.txt)."""
from pathlib import Path

JOBS = Path("/workspace/car-accident/stage2/aux_signal_experiments/jobs.txt")
G = "stage2/generalization"
SOURCES = ("AIHUB", "CCD", "MMAU", "NEXAR")
FOLDS = "stage2/long_context_v2_experiments/folds"


def loso(run, seeds, args, stop=9):
    """leave-one-source-out: train on the other sources, fixed stop epoch (the held-out source is never used for selection)."""
    return [f"{run}|{G}/results/{run}/{src}_seed{s}|--seed {s} --train-split {G}/loso/{src}_train.jsonl "
            f"--val-split {G}/loso/{src}_val.jsonl --stop-epoch {stop} {args}" for s in seeds for src in SOURCES]


def cv(run, seeds, args):
    return [f"{run}|{G}/results/{run}/cv/fold{k}_seed{s}|--seed {s} --train-split {FOLDS}/fold{k}_train.jsonl "
            f"--val-split {FOLDS}/fold{k}_val.jsonl {args}" for s in seeds for k in range(5)]


def add(lines, front=False):
    old = [l for l in JOBS.read_text().splitlines() if l.strip()] if JOBS.exists() else []
    hdr = [l for l in old if l.startswith("#")]; body = [l for l in old if not l.startswith("#")]
    body = (lines + body) if front else (body + lines)
    JOBS.write_text("\n".join(hdr + body) + "\n")
