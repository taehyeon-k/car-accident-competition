"""Map native sampled-index target mass to the augmented grid, without FPS."""
import numpy as np
import torch
import torch.nn.functional as F

def transport(base_frames, target_index, augmented_frames):
    base=np.asarray(base_frames);aug=np.asarray(augmented_frames)
    assert len(base)>0 and len(aug)>0
    assert np.all(np.diff(base)>=0) and np.all(np.diff(aug)>=0)
    mass=np.exp(-.5*(np.arange(len(base))-int(target_index))**2);mass/=mass.sum()
    if np.array_equal(base,aug):return mass.astype(np.float32)
    unique,inverse,counts=np.unique(aug,return_inverse=True,return_counts=True)
    right=np.searchsorted(unique,base).clip(0,len(unique)-1);left=(right-1).clip(0,len(unique)-1)
    nearest=np.where(np.abs(base-unique[left])<=np.abs(base-unique[right]),left,right)
    grouped=np.bincount(nearest,weights=mass,minlength=len(unique))
    # Same-frame positions share mass; do not favor an arbitrary duplicate slot.
    result=(grouped[inverse]/counts[inverse]).astype(np.float32)
    return result/result.sum()

def install(trainer):
    old_stride,old_collate,old_loss=trainer.stride_item,trainer.collate,trainer.total_loss
    def stride(base,k):
        item=old_stride(base,k)
        for event in ['entry','collision']:
            item['_transport_'+event]=transport(base['frame_numbers'].numpy(),base[event+'_index'],item['frame_numbers'].numpy())
        return item
    def collate(items,motion,lane):
        batch=old_collate(items,motion,lane)
        if any('_transport_entry' in item for item in items):
            length=batch['time_valid'].shape[1]
            for event in ['entry','collision']:
                targets=torch.zeros(len(items),length)
                for i,item in enumerate(items):
                    if '_transport_'+event in item:target=torch.from_numpy(item['_transport_'+event])
                    else:
                        target=torch.exp(-.5*(torch.arange(len(item['frame_numbers']))-item[event+'_index']).float().square());target/=target.sum()
                    targets[i,:len(target)]=target
                batch['transport_'+event]=targets
        return batch
    def loss(out,batch,cfg):
        if 'transport_entry' not in batch:return old_loss(out,batch,cfg)
        assert cfg['base_loss']=='nt' and not cfg['w_entry_aux'] and cfg['phase_rep']=='none' and cfg['boundary']=='none'
        terms=[]
        for event in ['entry','collision']:
            lp=F.log_softmax(out[event+'_logits'].float(),-1).masked_fill(~batch['time_valid'],0)
            terms.append(-(batch['transport_'+event]*lp).sum(-1).mean())
        direct=.5*(terms[0]+terms[1])
        side=F.cross_entropy(out['side_logits'].float(),batch['entry_side'])
        eva=F.binary_cross_entropy_with_logits(out['evasion_logits'].float(),batch['evasion'].float())
        return cfg['w_direct']*direct+.5*side+.5*eva,{'direct':direct}
    trainer.stride_item=stride;trainer.collate=collate;trainer.total_loss=loss
