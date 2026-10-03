"""Run the best Stage 3 trainer with a portable copy of the checkpoint configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/stage3/inference/code'))
if len(sys.argv) == 1:
    sys.argv += ['--config', str(ROOT / 'code/stage3/config.yaml')]
from stage3.run import main

if __name__ == '__main__': main()
