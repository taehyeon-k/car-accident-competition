"""Keep known duplicate components together, minimally changing existing folds.

Best-match audit is incomplete: this prevents KNOWN duplicates crossing folds,
not a certificate of complete deduplication. No feature extraction or training.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def main():
    old=ROOT/'stage2/long_context_v2_experiments/folds'
    rows={}; original={}
    for k in range(5):
        for line in (old/f'fold{k}_val.jsonl').read_text().splitlines():
            row=json.loads(line);sid=row['sample_id']
            assert sid not in rows
            rows[sid]=row;original[sid]=k
    parent={sid:sid for sid in rows}
    def find(s):
        while parent[s]!=s:
            parent[s]=parent[parent[s]];s=parent[s]
        return s
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    audit_path=ROOT/'stage2/generalization/results/dup_audit.json'
    audit=json.loads(audit_path.read_text());edges=[]
    for a,b,match,*_ in audit['lab_vs_lab']:
        if match>=.5:
            union(a,b);edges.append([a,b,match])
    paths={}
    for sid,row in rows.items():
        p=row.get('video_path')
        if p:
            if p in paths:union(sid,paths[p])
            paths[p]=sid
    groups=defaultdict(list)
    for sid in rows:groups[find(sid)].append(sid)
    assignment={};counts=Counter()
    for group,ids in sorted(groups.items(),key=lambda x:(-len(x[1]),x[0])):
        votes=Counter(original[sid] for sid in ids)
        chosen=min(votes,key=lambda k:(-votes[k],counts[k],k))
        for sid in ids:assignment[sid]=chosen
        counts[chosen]+=len(ids)
    out=OUT/'group_folds';out.mkdir(exist_ok=True)
    for k in range(5):
        for part in ['train','val']:
            selected=[rows[sid] for sid in sorted(rows) if (assignment[sid]==k)==(part=='val')]
            (out/f'fold{k}_{part}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    assert all(assignment[a]==assignment[b] for a,b,_ in edges)
    assert all(len({assignment[s] for s in ids})==1 for ids in groups.values())
    assert sum(counts.values())==349
    metadata={'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'rows':len(rows),'groups':len(groups),
        'nontrivial_groups':[sorted(ids) for ids in groups.values() if len(ids)>1],
        'fold_counts':dict(counts),'moved_clips':sum(assignment[s]!=original[s] for s in rows),
        'known_cross_fold_edges_before':sum(original[a]!=original[b] for a,b,_ in edges),'known_cross_fold_edges_after':0,
        'assignment':assignment,'group_by_id':{sid:find(sid) for sid in rows},
        'warning':'Best-match audit and exact video paths only; manually inspect uncertain graph edges. Existing CV checkpoints and old fold pseudo-labels are INVALID for evaluating these new folds. Refit baseline and candidate; regenerate expansion teachers/labels using new training partitions. Shared adapted-backbone provenance remains a separate caveat.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print({k:v for k,v in metadata.items() if k not in ['assignment','group_by_id','nontrivial_groups']})

if __name__=='__main__':main()
