"""Small exact-frame Stage-2 labeling server with autosave and resume."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_file


DEFAULT_ROOT = Path(os.environ.get("STAGE2_EXTERNAL_ROOT", "/workspace/data/stage2_external"))
REJECT_REASONS = [
    "ego_not_involved", "ego_changed_lane", "other_vehicle_not_intruding", "entry_not_visible",
    "collision_not_visible", "not_dashcam", "near_miss", "wrong_vehicle_type", "duplicate",
    "corrupt_video", "other",
]
HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-2 external labeling</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px system-ui}header{padding:10px 16px;background:#202020;display:flex;gap:16px;align-items:center}main{display:grid;grid-template-columns:minmax(560px,2fr) minmax(340px,1fr);gap:12px;padding:12px}.panel{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:12px}video,img{width:100%;max-height:45vh;background:#000;object-fit:contain}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}button,select,input,textarea{background:#2c2c2c;color:#eee;border:1px solid #555;border-radius:5px;padding:7px}button.active{background:#1769aa}button.bad{background:#8b2929}button.good{background:#276c38}.grow{flex:1}.value{font:700 16px ui-monospace}.meta{white-space:pre-wrap;max-height:30vh;overflow:auto;color:#bbb}kbd{background:#333;padding:2px 5px;border-radius:3px}.saved{color:#67d17a}</style></head>
<body><header><b>Stage-2 external labeling</b><span id="counter"></span><span id="status" class="saved"></span><span class="grow"></span><button onclick="previous()">P Previous</button><button onclick="next()">N Next</button></header>
<main><section class="panel"><video id="video" controls></video><div class="row"><button onclick="jumpAnchor()">J Anchor</button><button onclick="step(-10)">−10</button><button onclick="step(-5)">−5</button><button onclick="step(-1)">−1</button><span>position <span id="position" class="value"></span> / canonical frame <span id="canonical" class="value"></span></span><button onclick="step(1)">+1</button><button onclick="step(5)">+5</button><button onclick="step(10)">+10</button><button onclick="syncVideo()">Sync playback</button></div><img id="frame"><div class="row"><button onclick="mark('entry')">E Mark ENTRY</button><span id="entry" class="value"></span><button onclick="mark('collision')">C Mark COLLISION</button><span id="collision" class="value"></span></div></section>
<aside class="panel"><h2 id="sample"></h2><div class="row"><button id="usable1" class="good" onclick="setField('usable',true)">U USABLE</button><button id="usable0" class="bad" onclick="setField('usable',false)">X UNUSABLE</button></div><div class="row">entry side <button id="sideL" onclick="setField('entry_side','LEFT')">L LEFT</button><button id="sideR" onclick="setField('entry_side','RIGHT')">R RIGHT</button></div><div class="row">evasion space <button id="ev0" onclick="setField('evasion_space',0)">0</button><button id="ev1" onclick="setField('evasion_space',1)">1</button></div><div class="row"><label>reject reason <select id="reject" onchange="setField('reject_reason',this.value)"></select></label></div><textarea id="notes" rows="4" style="width:100%" placeholder="notes" oninput="noteChanged()"></textarea><div class="row"><button onclick="save()">S Save</button><span>Keys: ←/→ frame, Shift ±5, Ctrl ±10, E/C, L/R, 0/1, U/X, J, N/P, S</span></div><h3>Metadata evidence</h3><div id="meta" class="meta"></div></aside></main>
<script>
let queue=[],annotations={},index=0,pos=0,noteTimer=null;
const $=id=>document.getElementById(id);
async function init(){let state=await(await fetch('/api/state')).json();queue=state.queue;annotations=state.annotations;let pending=queue.findIndex(x=>!annotations[x.sample_id]);index=pending<0?0:pending;let sel=$('reject');sel.innerHTML='<option value=""></option>'+state.reject_reasons.map(x=>`<option>${x}</option>`).join('');load();}
function item(){return queue[index]} function ann(){return annotations[item().sample_id]||{sample_id:item().sample_id,source:item().source,source_id:item().source_id};}
function canonical(p=pos){return item().position_to_original_frame[Math.max(0,Math.min(item().position_to_original_frame.length-1,p))]}
function load(){let x=item(),a=ann();pos=a.last_position??x.metadata_collision_position??0;$('counter').textContent=`${index+1}/${queue.length}`;$('sample').textContent=x.sample_id;$('video').src=`/video/${encodeURIComponent(x.sample_id)}`;$('notes').value=a.notes||'';$('reject').value=a.reject_reason||'';$('meta').textContent=JSON.stringify({type:x.metadata_accident_type,collision_anchor:x.metadata_collision_frame_candidate,reason:x.filter_reason,positive:x.positive_evidence,negative:x.negative_evidence},null,2);refresh();}
function refresh(){let a=ann();$('position').textContent=pos;$('canonical').textContent=canonical();$('frame').src=`/frame/${encodeURIComponent(item().sample_id)}/${pos}?v=${Date.now()}`;$('entry').textContent=a.entry_frame??'—';$('collision').textContent=a.collision_frame??'—';for(let e of document.querySelectorAll('button.active'))e.classList.remove('active');if(a.usable===true)$('usable1').classList.add('active');if(a.usable===false)$('usable0').classList.add('active');if(a.entry_side==='LEFT')$('sideL').classList.add('active');if(a.entry_side==='RIGHT')$('sideR').classList.add('active');if(a.evasion_space===0)$('ev0').classList.add('active');if(a.evasion_space===1)$('ev1').classList.add('active');}
function step(n){pos=Math.max(0,Math.min(item().position_to_original_frame.length-1,pos+n));let a=ann();a.last_position=pos;annotations[item().sample_id]=a;refresh();}
function syncVideo(){pos=Math.max(0,Math.min(item().position_to_original_frame.length-1,Math.round($('video').currentTime*item().fps)));refresh();}
function jumpAnchor(){pos=item().metadata_collision_position;refresh()} function mark(k){let a=ann();a[k+'_position']=pos;a[k+'_frame']=canonical();annotations[item().sample_id]=a;refresh();save();}
function setField(k,v){let a=ann();a[k]=v;annotations[item().sample_id]=a;refresh();save()} function noteChanged(){clearTimeout(noteTimer);noteTimer=setTimeout(()=>{let a=ann();a.notes=$('notes').value;annotations[item().sample_id]=a;save()},500)}
async function save(){let a=ann();a.notes=$('notes').value;a.updated_at=new Date().toISOString();annotations[item().sample_id]=a;$('status').textContent='saving…';let r=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(a)});$('status').textContent=r.ok?'saved':'save failed';}
async function next(){await save();index=Math.min(queue.length-1,index+1);load()} async function previous(){await save();index=Math.max(0,index-1);load()}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;let d=e.ctrlKey?10:e.shiftKey?5:1;if(e.key==='ArrowLeft'){e.preventDefault();step(-d)}else if(e.key==='ArrowRight'){e.preventDefault();step(d)}else if(e.key==='e')mark('entry');else if(e.key==='c')mark('collision');else if(e.key==='l')setField('entry_side','LEFT');else if(e.key==='r')setField('entry_side','RIGHT');else if(e.key==='0')setField('evasion_space',0);else if(e.key==='1')setField('evasion_space',1);else if(e.key==='u')setField('usable',true);else if(e.key==='x')setField('usable',false);else if(e.key==='j')jumpAnchor();else if(e.key==='n')next();else if(e.key==='p')previous();else if(e.key==='s')save();});init();
</script></body></html>"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_app(root: Path) -> Flask:
    app = Flask(__name__)
    queue_path = root / "labeling/queue.jsonl"
    annotations_path = root / "labeling/annotations.jsonl"
    lock = threading.Lock()
    queue = load_jsonl(queue_path)
    by_id = {row["sample_id"]: row for row in queue}

    def annotations() -> dict[str, dict[str, Any]]:
        return {row["sample_id"]: row for row in load_jsonl(annotations_path)}

    def persist(values: dict[str, dict[str, Any]]) -> None:
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=annotations_path.parent, delete=False) as stream:
            for sample_id in sorted(values):
                stream.write(json.dumps(values[sample_id], ensure_ascii=False, sort_keys=True) + "\n")
            temporary = Path(stream.name)
        temporary.replace(annotations_path)

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(queue=queue, annotations=annotations(), reject_reasons=REJECT_REASONS)

    @app.post("/api/annotation")
    def save_annotation() -> Response:
        record = request.get_json(force=True)
        sample_id = record.get("sample_id")
        if sample_id not in by_id:
            abort(400)
        with lock:
            values = annotations()
            values[sample_id] = record
            persist(values)
        return jsonify(ok=True)

    @app.get("/video/<sample_id>")
    def video(sample_id: str):
        if sample_id not in by_id:
            abort(404)
        return send_file(by_id[sample_id]["playback_video_path"], conditional=True)

    @app.get("/frame/<sample_id>/<int:position>")
    def frame(sample_id: str, position: int):
        item = by_id.get(sample_id)
        if item is None or not 0 <= position < len(item["position_to_original_frame"]):
            abort(404)
        process = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", item["original_video_path"],
                "-vf", f"select=eq(n\\,{position})", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if process.returncode or not process.stdout:
            abort(500)
        return send_file(io.BytesIO(process.stdout), mimetype="image/jpeg", max_age=3600)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17070)
    arguments = parser.parse_args()
    create_app(arguments.root).run(host=arguments.host, port=arguments.port, threaded=True)


if __name__ == "__main__":
    main()
