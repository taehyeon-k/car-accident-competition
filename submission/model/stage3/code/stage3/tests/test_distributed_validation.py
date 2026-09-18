import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_process_validation_matches_single_process_without_padding_duplicates(tmp_path):
    env = {**os.environ, 'STAGE3_DIST_OUTPUT': str(tmp_path), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '-m', 'stage3.tests.distributed_validation_worker']
    completed = subprocess.run(command, env=env, cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((tmp_path / 'rank-0.json').read_text()) == json.loads((tmp_path / 'rank-1.json').read_text())
