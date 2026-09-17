import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rows=[r for r in json.loads((p/'history.json').read_text()) if 'train_config/epoch' in r]
x=[r['train_config/epoch'] for r in rows]
keys=sorted(set().union(*(r.keys() for r in rows)))
with (p/'epochs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for key,label in [('train/competition_score','Train (augmented crops)'),('val/competition_score','Validation (full videos)')]:
 axes[0,0].plot(x,[r[key] for r in rows],label=label,linewidth=2)
axes[0,0].set_title('Training score approaches 1; validation plateaus')
for key,label in [('acc_entry_0.3s','ENTRY'),('acc_collision_0.3s','COLLISION'),('f1_entry_side_macro','Entry side F1'),('f1_evasion_space_macro','Evasion F1')]:
 axes[0,1].plot(x,[r['val/'+key] for r in rows],label=label,linewidth=1.8)
axes[0,1].set_title('Validation: ENTRY and evasion need attention')
for s in ['AIHUB','CCD','NEXAR']:
 axes[1,0].plot(x,[r[f'val/{s}/acc_entry_0.3s'] for r in rows],label=s,linewidth=2)
axes[1,0].set_title('NEXAR ENTRY remains weak throughout')
for key,label in [('val/loss_evasion_space','Evasion validation loss'),('val/loss_entry_side','Side validation loss')]:
 axes[1,1].plot(x,[r[key] for r in rows],label=label,linewidth=2)
axes[1,1].set_title('Evasion loss rises after early epochs')
for ax in axes.flat:
 ax.axvline(13,color='gray',linestyle=':',alpha=.8,label='Best checkpoint' if ax==axes[0,0] else None)
 ax.set_xlabel('Epoch');ax.set_xticks([1,4,7,10,13,16,19]);ax.grid(alpha=.2);ax.legend(fontsize=8)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_ylim(0,1.04)
fig.suptitle('Recent run kqr8dczz | 201 train / 50 validation | best score 0.6616 at epoch 13',fontsize=14)
fig.savefig(p/'learning_curves.png',dpi=170)
fig.savefig(p/'learning_curves.pdf')
print('Wrote epochs.csv, learning_curves.png, learning_curves.pdf')
