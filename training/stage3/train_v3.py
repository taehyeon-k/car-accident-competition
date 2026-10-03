"""Run the archived V3 trainer with a portable copy of the checkpoint configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model/stage3/code'))
if len(sys.argv) == 1:
    sys.argv += ['--config', str(ROOT / 'training/stage3/v3.yaml')]
from stage3.run import main

if __name__ == '__main__': main()
