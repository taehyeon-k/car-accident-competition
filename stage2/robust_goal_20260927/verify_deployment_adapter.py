"""Strict-load and actual cached-clip parity for the isolated sharp deployment adapter."""
import json
import sys
from pathlib import Path
import torch
from stage2.generalization import robust_eval as R
from stage2.long_context_v2_experiments import common as C
from stage2.robust_goal_20260927.model import build
from stage2.robust_goal_20260927.resource_guard import resource_guard

OUT=Path(__file__).resolve().parent


def main():
    resource_guard(10.5)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    sys.path.insert(0,str((OUT/'deployment_adapter').resolve()))
    import runtime
    folder=OUT/'training/aux_sharp/cv/fold0_seed0'
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['config']
    trained=build(cfg).eval();trained.load_state_dict(checkpoint['model'],strict=True)
    adapted=runtime.build_member({'arch':'aux','motion':'both','motion_dim':cfg['motion_dim'],
                                  'boundary':'none','entry_aux':True}).eval()
    adapted.load_state_dict(checkpoint['model'],strict=True)
    by_source={}
    for row in C.rows(str(OUT/'group_folds/fold0_val.jsonl')):
        source=row['source_id'].split(':')[0]
        if source in ('AIHUB','CCD','MMAU','NEXAR') and source not in by_source:
            by_source[source]=row
    assert len(by_source)==4
    reference=json.loads((folder/'rate_eval.json').read_text())
    saved={rate:{p['sample_id']:p for p in reference[rate]['predictions']} for rate in ('k1','k2','k3')}
    result={'checkpoint':str(folder/'checkpoint.pt'),'clips':[],'max_abs_difference':{}}
    for source,row in sorted(by_source.items()):
        for k in (1,2,3):
            resource_guard(10.5)
            item=R.item(row,k)
            x=item['x'][None];valid=torch.ones((1,len(item['frames'])),dtype=torch.bool)
            motion=item['both'][None]
            with torch.inference_mode():
                a=trained(x,valid,motion=motion)
                b=adapted(x,valid,motion=motion)
            record={'source':source,'sample_id':row['sample_id'],'rate':k,'positions':len(item['frames'])}
            for key in ('entry_logits','collision_logits','side_logits','evasion_logits'):
                difference=float((a[key]-b[key]).abs().max())
                result['max_abs_difference'][key]=max(result['max_abs_difference'].get(key,0.0),difference)
                assert torch.equal(a[key],b[key]),(source,k,key,difference)
            prediction=R.predict([(adapted,'both')],item,torch.device('cpu'))
            expected=saved[f'k{k}'][row['sample_id']]
            for key in ('entry_frame','collision_frame','entry_side','evasion_space'):
                assert prediction[key]==expected[key],(source,k,key,prediction[key],expected[key])
            result['clips'].append(record)
            C.dense.cache_clear()
    result['status']='strict_load_all_outputs_and_12_cached_clip_predictions_exact'
    (OUT/'deployment_adapter_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],result['max_abs_difference'])


if __name__=='__main__':main()
