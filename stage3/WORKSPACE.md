# Workspace integration

The prepared container is `car-accident-dev`, with the host workspace mounted at
`/workspace`. It runs Python 3.11, PyTorch 2.8.0+cu128, and exposes the NVIDIA
L40S. Start it when needed:

```bash
docker start car-accident-dev
docker exec -it -w /workspace/car-accident car-accident-dev bash
```

BATON is at `/workspace/data/stage3/BATON-Sample`. The adapter maps `time_s`,
`vEgo`, `aEgo`, and `steeringAngleDeg`. The 100 Hz sensor rows are interpolated
to actual selected video timestamps. Video is 20 Hz in this sample and is
PTS-resampled to 10 Hz for training.

Official SEA-RAFT assets are local and inference performs no downloads:

- source: `/workspace/pretrained/sea_raft/source`
- source revision: `9137517ba24e628442aec097d3afe71d03503b75`
- S checkpoint: `/workspace/pretrained/sea_raft/model.safetensors`
- checkpoint SHA-256: `9752a49fca0a3f33551d51fec7bc5bd87f93a155b1e9fadd803e8176f4edb423`

The official constructor normally initializes from torchvision ImageNet weights.
The wrapper disables that download because the SEA-RAFT checkpoint replaces the
learned tensors. It strictly permits only the checkpoint's documented shared
`bn3`/`downsample.1` safetensors aliases to be absent.

The current 18-hour BATON sample produces 2,161 thirty-second segments. Schema-2
cache compression is estimated near 18 GiB; verify free space before a full cache
run. The checked smoke caches and generated run artifacts live below
`/workspace/cache/stage3` and `/workspace/car-accident/runs/stage3`.

ADAS-TO is intentionally not guessed. Its adapter raises until a concrete release
provides timestamp, direct longitudinal acceleration, steering definition/unit,
and sign mappings.
