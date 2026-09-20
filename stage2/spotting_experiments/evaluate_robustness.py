from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SpottingFeatures, collate, read_rows
from .models import build_model
from .train import move, normalized_metrics, official_metrics, predict


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--counts",type=int,nargs="+",default=[96,128,160]); args=p.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=state["config"]; rows=read_rows(cfg["val_manifest"]); all_rows=read_rows(cfg["all_manifest"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_model(cfg).to(device).eval(); model.load_state_dict(state["model"])
    results={}; by_id={}
    for count in args.counts:
        ds=SpottingFeatures(rows,cfg["cache_dir"],sample_counts=[count],strict_fps_blind=True); loader=DataLoader(ds,batch_size=cfg.get("eval_batch_size",16),collate_fn=collate)
        predictions,_=predict(model,loader,device); results[str(count)]={"fpsblind":normalized_metrics(predictions),"official_offline":official_metrics(predictions,all_rows)}
        for x in predictions: by_id.setdefault(x["sample_id"],[]).append(x)
    entry_std=[np.std([x["entry_frame"] for x in values])/max(next(r["num_frames"] for r in rows if r["sample_id"]==sid)-1,1) for sid,values in by_id.items()]
    collision_std=[np.std([x["collision_frame"] for x in values])/max(next(r["num_frames"] for r in rows if r["sample_id"]==sid)-1,1) for sid,values in by_id.items()]
    results["prediction_variance"]={"entry_normalized_std_mean":float(np.mean(entry_std)),"collision_normalized_std_mean":float(np.mean(collision_std))}
    fused=[]
    for views in by_id.values():
        item=dict(views[0])
        item["entry_frame"]=int(round(np.median([x["entry_frame"] for x in views])))
        item["collision_frame"]=int(round(np.median([x["collision_frame"] for x in views])))
        item["entry_frame"]=min(item["entry_frame"],item["collision_frame"])
        item["entry_side"]=int(np.mean([x["entry_side"] for x in views])>=.5)
        item["evasion_space"]=int(np.mean([x["evasion_space"] for x in views])>=.5)
        fused.append(item)
    results["ensemble_median"]={"fpsblind":normalized_metrics(fused),"official_offline":official_metrics(fused,all_rows),"views":args.counts}
    Path(args.output).write_text(json.dumps(results,indent=2)+"\n"); print(json.dumps(results,indent=2))
if __name__=="__main__":main()
