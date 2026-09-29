"""Real cached-data forward/backward check for the prepared transport arm."""
import json
from pathlib import Path
from .resource_guard import resource_guard

# Admission before importing torch keeps a waiting check from reserving RAM.
if __name__=='__main__':resource_guard(12.5)

import torch
from . import train_cpu as T
from .model import build
from .transport_targets import install
from stage2.long_context_v2_experiments import common as C

OUT=Path(__file__).resolve().parent

def main():
    assert not torch.cuda.is_available();torch.set_num_threads(1);resource_guard(10.5)
    cfg=json.loads((OUT/'training/base/cv/fold0_seed0/config.json').read_text())
    row=next(r for r in C.rows(str(OUT/'group_folds/fold0_train.jsonl')) if C.source(r)=='CCD')
    base=T.attach_inputs(T.lazy_item(row),'both','none',False)
    old_loss=T.total_loss;install(T)
    aug=T.stride_item(base,3)
    native=T.collate([T.materialize_any(base)],'custom','none')
    mixed=T.collate([T.materialize_any(base),T.materialize_any(aug)],'custom','none')
    for e in ['entry','collision']:
        assert torch.allclose(mixed['transport_'+e].sum(-1),torch.ones(2),atol=1e-6)
        assert not (mixed['transport_'+e][~mixed['time_valid']]!=0).any()
    torch.manual_seed(0);model=build(cfg).eval()
    raw=model(native['x'],native['time_valid'],motion=native['motion'])
    a,_=old_loss(raw,native,cfg);b,_=T.total_loss(raw,native,cfg)
    assert torch.equal(a,b)
    raw=model(mixed['x'],mixed['time_valid'],motion=mixed['motion'])
    loss,_=T.total_loss(raw,mixed,cfg);loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    record={'sample_id':row['sample_id'],'native_loss_exactly_unchanged':True,'mixed_view_target_mass_and_padding':'pass','mixed_view_loss':float(loss.detach()),'finite_backward':'pass','grad_tensors':len(grads)}
    (OUT/'transport_integration_check.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)

if __name__=='__main__':main()
