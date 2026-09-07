"""Deterministic Hungarian association for persistent object tracks."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
@dataclass
class Detection: box: np.ndarray; score: float; label: str
@dataclass
class Track:
    id: int; label: str; observations: dict[int, Detection] = field(default_factory=dict); last_t: int = -1

def iou(a: np.ndarray, b: np.ndarray) -> float:
    wh = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])); inter = wh[0]*wh[1]
    return float(inter / max(1e-8, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter))
def center_distance(a: np.ndarray, b: np.ndarray, diagonal: float) -> float:
    return float(np.linalg.norm((a[:2]+a[2:])/2 - (b[:2]+b[2:])/2) / max(diagonal, 1e-8))

class HungarianTracker:
    def __init__(self, max_gap: int=2, max_center_distance: float=.20, max_match_cost: float=.65):
        self.max_gap, self.max_center_distance, self.max_match_cost = max_gap, max_center_distance, max_match_cost
    def track(self, frames: list[list[Detection]], sizes: list[tuple[int,int]]) -> list[Track]:
        tracks: list[Track] = []; next_id = 0
        for t, detections in enumerate(frames):
            live = [x for x in tracks if t-x.last_t <= self.max_gap+1]
            candidates = [(x, x.observations[x.last_t]) for x in live]
            cost = np.full((len(candidates),len(detections)), 1e6, dtype=np.float64)
            diag = float(np.hypot(*sizes[t]))
            for r,(track,prev) in enumerate(candidates):
                for c,det in enumerate(detections):
                    if track.label != det.label: continue
                    dist = center_distance(prev.box,det.box,diag); value=.6*(1-iou(prev.box,det.box))+.4*dist
                    if dist <= self.max_center_distance and value <= self.max_match_cost: cost[r,c]=value
            used=set()
            if cost.size:
                rows, cols = linear_sum_assignment(cost)
                for r,c in zip(rows,cols):
                    if cost[r,c] >= 1e6: continue
                    track,_=candidates[r]; track.observations[t]=detections[c]; track.last_t=t; used.add(c)
            for c,d in enumerate(detections):
                if c not in used:
                    tracks.append(Track(next_id,d.label,{t:d},t)); next_id+=1
        return tracks

def rank_tracks(tracks: list[Track], total_frames: int, sizes: list[tuple[int,int]], loom_clip: float=np.log(2), max_tracks: int=12) -> list[Track]:
    def score(track: Track) -> float:
        areas=[]; bottoms=[]; conf=[]
        for t,d in track.observations.items():
            w,h=sizes[t]; box=d.box; areas.append((box[2]-box[0])*(box[3]-box[1])/(w*h)); bottoms.append(box[3]/h); conf.append(d.score)
        deltas=np.maximum(0,np.diff(np.log(np.asarray(areas)+1e-8))) if len(areas)>1 else np.zeros(1)
        loom=float(np.clip(deltas.max(initial=0)/loom_clip,0,1))
        return .30*max(areas)+.20*max(bottoms)+.20*(len(areas)/total_frames)+.10*float(np.mean(conf))+.20*loom
    return sorted(tracks,key=score,reverse=True)[:max_tracks]
