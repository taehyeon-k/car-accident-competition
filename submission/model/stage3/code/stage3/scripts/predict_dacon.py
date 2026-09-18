from __future__ import annotations

import argparse

from stage3.inference.dacon import predict_stage3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DACON Stage 3 inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = predict_stage3(args.data_dir, args.model_dir)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
