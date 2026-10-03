"""Offline DACON entry points; all executable assets are relative to this file."""
from pathlib import Path
import gc
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent


def _stage_dir(model_dir, stage):
    path = Path(model_dir).expanduser().resolve()
    return path / stage if (path / stage).is_dir() else path


def _load(stage):
    name = '_dacon_' + stage
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'model' / stage / 'runtime.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_stage1(data_dir, model_dir):
    try:
        return _load('stage1').predict_stage1(data_dir, _stage_dir(model_dir, 'stage1'))
    finally:
        _release()


def predict_stage2(data_dir, model_dir):
    try:
        return _load('stage2').predict(data_dir, _stage_dir(model_dir, 'stage2'))
    finally:
        _release()


def predict_stage3(data_dir, model_dir):
    try:
        return _load('stage3').predict(data_dir, _stage_dir(model_dir, 'stage3'))
    finally:
        _release()
