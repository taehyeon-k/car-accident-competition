from __future__ import annotations

import argparse

import numpy as np

from stage3.data.adapters.baton import BatonAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect canonical BATON signal mappings")
    parser.add_argument("--data-root", default="/workspace/data/stage3/BATON-Sample")
    args = parser.parse_args()
    records = BatonAdapter(args.data_root).discover()
    for record in records:
        signals = record.signals
        print(
            record.clip_id,
            f"duration={signals.t[-1] - signals.t[0]:.1f}s",
            f"samples={len(signals.t)}",
            f"speed=[{np.nanmin(signals.v):.2f},{np.nanmax(signals.v):.2f}]m/s",
            f"accel=[{np.nanmin(signals.a_long):.2f},{np.nanmax(signals.a_long):.2f}]m/s2",
            f"steer=[{np.nanmin(signals.steering_angle):.2f},{np.nanmax(signals.steering_angle):.2f}]deg",
        )


if __name__ == "__main__":
    main()
