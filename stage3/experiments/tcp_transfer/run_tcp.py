"""Run the frozen TCP feature screen using the normal Stage 3 experiment report."""
from stage3.experiments import run

from .trainer import TCPTrainer

run.Trainer = TCPTrainer

if __name__ == '__main__':
    run.main()
