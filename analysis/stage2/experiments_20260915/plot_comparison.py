import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/accident-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parent
summary = json.loads((out / 'summary.json').read_text())
replay = json.loads((out / 'decoder_replay.json').read_text())
names = ['baseline', 'span_lr', 'span_lr_dr']
labels = ['Baseline', 'Lower LR + broader targets', 'More dropout + LR/target changes']
colors = ['#506579', '#de8331', '#137f78']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
for name, label, color in zip(names, labels, colors):
    rows = [r for r in json.loads((out / f'{name}_history.json').read_text())
            if 'val/competition_score' in r]
    x = [r['train_config/epoch'] for r in rows]
    axes[0, 0].plot(x, [r['val/competition_score'] for r in rows], label=label, color=color)
    axes[0, 1].plot(x, [r['train/competition_score'] for r in rows], label=label, color=color)
    axes[1, 0].plot(x, [r['val/loss_evasion_space'] for r in rows], label=label, color=color)
axes[0, 0].set_title('Validation gains are modest and noisy')
axes[0, 0].set_ylim(.38, .72)
axes[0, 1].set_title('All three runs nearly fit the training set perfectly')
axes[1, 0].set_title('Higher dropout has not stopped evasion loss rising')
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax.set_xticks([1, 4, 7, 10, 13, 16, 19])
modes = ['baseline_fp32', 'span200_fp32', 'tolerance_mass_unlimited', 'tolerance_mass_200frames']
bars = [replay[m]['metrics']['score'] for m in modes]
axes[1, 1].barh(np.arange(4), bars, color=['#506579', '#889dac', '#65aaa5', '#137f78'])
axes[1, 1].set_yticks(np.arange(4), ['Original decode', '+ 200-frame cap', 'Tolerance mass*', 'Tolerance mass* + cap'])
axes[1, 1].invert_yaxis(); axes[1, 1].set_xlim(0, .82)
for i, value in enumerate(bars):
    axes[1, 1].text(value + .008, i, f'{value:.4f}', va='center')
axes[1, 1].set_title('Same baseline logits: decoding-only experiments')
axes[1, 1].set_xlabel('Validation score; *requires known FPS for ±0.3 s')
fig.suptitle('Stage 2 experiments — September 15 | dropout run observed through epoch 14', fontsize=14)
fig.savefig(out / 'comparison.png', dpi=170)
fig.savefig(out / 'comparison.pdf')
print('Saved comparison.png and comparison.pdf')
