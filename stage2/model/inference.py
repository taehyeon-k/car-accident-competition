"""Native-frame Stage 2 inference with independent tracks and shared frame caches."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode

from stage2.data.cache_geometry import frame_paths, compact_observations
from stage2.data.preparation import build_window_geometry, load_observation
from stage2.data.sampling import build_coarse_bins, recover_region, sliding_windows
from stage2.data.transforms import letterbox
from stage2.model.backbones import FrozenAdapter
from stage2.test import merge_window_logits, side_label, evasion_label


class Stage2Pipeline:
    """Run coarse localization then native-frame refinement for both events.

    Per-frame RF-DETR/depth scalars are read from disk when supplied. Without a
    geometry_dir, the configured local frozen adapters compute only needed frames.
    Dense DINO grids live on CPU between windows and are released at their last
    use. Coarse and fine visual modules are moved to the device sequentially.
    """

    def __init__(
        self,
        coarse,
        fine,
        config: dict,
        device: str = "cpu",
    ):
        self.coarse = coarse.eval().cpu()
        self.fine = fine.eval().cpu()
        self.config = config
        self.device = torch.device(device)
        self.detector = None
        self.depth = None

    def _records(
        self,
        row,
        paths,
        ids,
        positions,
        cache,
        images,
    ):
        missing = sorted(set(int(position) for position in positions) - cache.keys())
        if row.get("geometry_dir"):
            for position in missing:
                cache[position] = load_observation(
                    row["geometry_dir"],
                    ids[position],
                )
            return
        if not missing:
            return

        self.fine.cpu()

        model_config = self.config["model"]
        orientation = model_config.get("depth_closer_is_larger")
        if not isinstance(
            orientation,
            bool,
        ):
            raise ValueError(
                "Verify and configure depth_closer_is_larger before inference"
            )
        if self.detector is None:
            self.detector = FrozenAdapter(
                model_config["rfdetr_factory"],
                model_config["rfdetr_checkpoint"],
            )
            self.depth = FrozenAdapter(
                model_config["depth_factory"],
                model_config["depth_checkpoint"],
            )

        self.detector.to(self.device)
        self.depth.to(self.device)
        chunk_size = model_config.get(
            "geometry_batch_size",
            4,
        )
        for start in range(
            0,
            len(missing),
            chunk_size,
        ):
            chunk = missing[start : start + chunk_size]
            for position in chunk:
                if position not in images:
                    images[position] = read_image(
                        str(paths[position]),
                        mode=ImageReadMode.RGB,
                    )
            rgb = [images[position].to(self.device) for position in chunk]
            detections = self.detector(rgb)
            depths = self.depth(rgb)
            if len(detections) != len(chunk) or len(depths) != len(chunk):
                raise ValueError(
                    "Frozen adapters must return one observation per frame"
                )
            for position, image, detected, depth in zip(
                chunk,
                rgb,
                detections,
                depths,
            ):
                cache[position] = compact_observations(
                    detected,
                    depth,
                    image.shape[-1],
                    image.shape[-2],
                    self.config["tracking"]["detection_threshold"],
                    orientation,
                )
        self.detector.cpu()
        self.depth.cpu()

    def _batch(
        self,
        item: dict,
    ) -> dict:
        return {
            name: value.unsqueeze(0).to(self.device) for name, value in item.items()
        }

    @torch.inference_mode()
    def predict(
        self,
        row: dict,
    ) -> dict:
        paths, ids = frame_paths(row["frames_dir"])
        bins = build_coarse_bins(
            list(range(len(ids))),
            ids,
            training=False,
            num_bins=getattr(self.coarse, "num_frames", 32),
        )
        records = {}
        images = {}
        positions = bins.representative_native_pos
        self._records(
            row,
            paths,
            ids,
            positions,
            records,
            images,
        )
        for position in np.unique(positions):
            if int(position) not in images:
                images[int(position)] = read_image(
                    str(paths[position]),
                    mode=ImageReadMode.RGB,
                )
        coarse_rgb = {
            position: letterbox(
                image,
                384,
            )[0]
            for position, image in images.items()
        }
        item = build_window_geometry(
            [records[int(position)] for position in positions],
            bins.valid,
            self.config["tracking"],
            coarse=True,
        )
        item["bin_valid"] = torch.from_numpy(bins.valid)
        item["coarse_rgb"] = torch.stack(
            [coarse_rgb[int(position)] for position in positions]
        ).permute(
            1,
            0,
            2,
            3,
        )

        self.coarse.to(self.device)
        coarse_outputs = {
            key: value.cpu() for key, value in self.coarse(self._batch(item)).items()
        }
        self.coarse.cpu()
        del coarse_rgb, item

        jobs = []
        for event in ("entry", "collision"):
            predicted_bin = int(coarse_outputs[f"{event}_logits"].argmax(-1).item())
            region = recover_region(
                bins,
                predicted_bin,
            )
            jobs.extend((event, window) for window in sliding_windows(region))
        jobs.sort(key=lambda job: int(job[1][0]))
        uses = Counter(int(position) for _, window in jobs for position in window)
        for position in list(images):
            if position not in uses:
                images.pop(position)

        feature_cache = {}
        event_windows = {"entry": [], "collision": []}
        event_logits = {"entry": [], "collision": []}
        for event, window in jobs:
            # The frozen adapters run before moving fine visual weights to GPU.
            self._records(
                row,
                paths,
                ids,
                window,
                records,
                images,
            )
            valid = np.arange(64) < len(window)
            padded_positions = np.pad(
                window,
                (0, 64 - len(window)),
                mode="edge",
            )
            item = build_window_geometry(
                [records[int(position)] for position in padded_positions],
                valid,
                self.config["tracking"],
                coarse=False,
            )
            item["time_valid"] = torch.from_numpy(valid)
            needed = [
                int(position)
                for position in window
                if int(position) not in feature_cache
            ]
            self.fine.to(self.device)
            for start in range(
                0,
                len(needed),
                self.fine.frame_batch_size,
            ):
                chunk = needed[start : start + self.fine.frame_batch_size]
                rgb = []
                for position in chunk:
                    if position not in images:
                        images[position] = read_image(
                            str(paths[position]),
                            mode=ImageReadMode.RGB,
                        )
                    rgb.append(
                        letterbox(
                            images[position],
                            336,
                        )[0]
                    )
                global_features, dense_features = self.fine.visual(
                    torch.stack(rgb).to(self.device)
                )
                for index, position in enumerate(chunk):
                    feature_cache[position] = (
                        global_features[index].cpu(),
                        dense_features[index].cpu(),
                    )

            first_global, first_dense = feature_cache[int(window[0])]
            globals_padded = first_global.new_zeros(
                64,
                first_global.shape[-1],
            )
            dense_padded = first_dense.new_zeros(
                64,
                24,
                24,
                first_dense.shape[-1],
            )
            for index, position in enumerate(window):
                globals_padded[index], dense_padded[index] = feature_cache[
                    int(position)
                ]
            batch = self._batch(item)
            output = self.fine.head(
                globals_padded[None].to(self.device),
                dense_padded[None].to(self.device),
                batch["boxes_grid"],
                batch["geometry"],
                batch["object_valid"],
                batch["time_valid"],
            )
            event_windows[event].append(
                np.asarray([ids[position] for position in window])
            )
            event_logits[event].append(
                output[f"{event}_logits"][0, : len(window)].cpu().numpy()
            )
            for position in window:
                position = int(position)
                uses[position] -= 1
                if uses[position] == 0:
                    feature_cache.pop(position)
                    images.pop(
                        position,
                        None,
                    )

        self.fine.cpu()

        prediction = {
            "entry_frame": merge_window_logits(
                event_windows["entry"],
                event_logits["entry"],
            )[0],
            "collision_frame": merge_window_logits(
                event_windows["collision"],
                event_logits["collision"],
            )[0],
            "entry_side": side_label(coarse_outputs["direction_logits"][0].numpy()),
            "evasion_space": evasion_label(
                float(coarse_outputs["evasion_logits"].item())
            ),
        }
        if (
            prediction["entry_frame"] not in ids
            or prediction["collision_frame"] not in ids
        ):
            raise RuntimeError("Submission prediction is not an original filename ID")
        return prediction
