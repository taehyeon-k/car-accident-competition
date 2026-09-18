from __future__ import annotations

import argparse

import torch

from stage3.inference.predictor import Stage3Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage 3 inference by pipeline section")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    predictor = Stage3Predictor(args.checkpoint, args.device)
    if predictor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(predictor.device)
    acceleration, steering, timing = predictor.predict_video(args.video, args.max_frames)
    for name, seconds in timing.items():
        print(f"{name:16s} {seconds:9.3f}s")
    print(f"rows             {len(acceleration)}")
    if predictor.device.type == "cuda":
        print(f"cuda_allocated_gib {torch.cuda.memory_allocated(predictor.device) / 2**30:.3f}")
        print(f"cuda_peak_gib      {torch.cuda.max_memory_allocated(predictor.device) / 2**30:.3f}")


if __name__ == "__main__":
    main()
