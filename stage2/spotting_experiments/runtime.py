"""Private-style FPS-blind inference for experiment checkpoints."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stage2.geometry_pretrain.models.geometry_dino import DinoBackbone
from .extract_features import Images, indexed_frames, MEAN, STD
from .models import build_model
from .objective import decode
from .sampling import normalized_indices

@torch.inference_mode()
def encode(backbone, paths, device, batch_size=24, workers=4):
    result=[]
    for images in DataLoader(Images(paths),batch_size=batch_size,num_workers=workers,pin_memory=device.type=="cuda"):
        images=(images.to(device,non_blocking=True).float()-MEAN.to(device))/STD.to(device)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            patch=backbone(images)["patch"]
        result.append(F.adaptive_avg_pool2d(patch.float(),(7,10)).flatten(2).transpose(1,2).half())
    return torch.cat(result)[None]

@torch.inference_mode()
def predict(data_dir, head_checkpoint, backbone_checkpoint, sample_count=128):
    state=torch.load(head_checkpoint,map_location="cpu",weights_only=False); cfg=state["config"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone=DinoBackbone("vits16",backbone_checkpoint).to(device).eval()
    head=build_model(cfg).to(device).eval(); head.load_state_dict(state["model"])
    root=Path(data_dir); root=root/"stage2" if (root/"stage2").is_dir() else root; root=root/"images" if (root/"images").is_dir() else root
    rows=[]; timing={"backbone_seconds":0.,"head_seconds":0.,"encoded_frames":0}
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        pairs=indexed_frames(folder); chosen=normalized_indices([x[0] for x in pairs],sample_count)
        unique,inverse=np.unique(chosen,return_inverse=True); paths=[pairs[i][1] for i in unique]
        if device.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter(); features=encode(backbone,paths,device)
        if device.type=="cuda": torch.cuda.synchronize()
        timing["backbone_seconds"]+=time.perf_counter()-start; timing["encoded_frames"]+=len(unique)
        features=features[:,torch.as_tensor(inverse,device=device)]; frames=torch.tensor([[pairs[i][0] for i in chosen]],device=device)
        batch={"frame_numbers":frames,"normalized_positions":torch.linspace(0,1,sample_count,device=device)[None],"time_valid":torch.ones(1,sample_count,dtype=torch.bool,device=device)}
        start=time.perf_counter(); out=head(features,batch["time_valid"]); entry,collision=decode(out,batch)
        if device.type=="cuda": torch.cuda.synchronize()
        timing["head_seconds"]+=time.perf_counter()-start
        rows.append({"ID":folder.name,"collision_frame":int(collision),"entry_frame":int(entry),"evasion_space":int(out["evasion_logits"].item()>=0),"entry_side":["LEFT","RIGHT"][int(out["side_logits"].argmax())]})
    timing["videos"]=len(rows); timing["peak_vram_mb"]=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0
    return pd.DataFrame(rows),timing

def main():
    p=argparse.ArgumentParser(); p.add_argument("data_dir"); p.add_argument("--checkpoint",required=True); p.add_argument("--backbone-checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--runtime-json"); p.add_argument("--sample-count",type=int,default=128); a=p.parse_args()
    frame,timing=predict(a.data_dir,a.checkpoint,a.backbone_checkpoint,a.sample_count); frame.to_csv(a.output,index=False)
    if a.runtime_json: Path(a.runtime_json).write_text(json.dumps(timing,indent=2)+"\n")
    print(frame); print(json.dumps(timing,indent=2))
if __name__=="__main__":main()
