"""Plot saved evidence without rereading changing experiment results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
rows = json.loads((out/'leaderboard_correlations.json').read_text())['rows']
fig, axs = plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
for ax,key,title in zip(axs,['fixed_all','fixed_nexar','cv_all'],['Fixed overall vs leaderboard (n=7)','Fixed NEXAR vs leaderboard (n=7)','CV overall vs leaderboard (n=4)']):
    for row in rows:
        if row.get(key) is None:
            continue
        label = row['name'].split()[0]
        label = {'NEXAR':'specialist','Length':'hybrid'}.get(label,label)
        color = '#15803d' if label == 'v8' else '#d97706' if label in ['specialist','hybrid'] else '#2563eb'
        ax.scatter(row[key],row['leaderboard'],s=65,color=color)
        ax.annotate(label,(row[key],row['leaderboard']),xytext=(5,5),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Offline score / recipe proxy',ylabel='Stage 2 leaderboard score',title=title)
    ax.grid(alpha=.2)
    ax.set_ylim(.395,.615)
    ax.margins(x=.2)
fig.suptitle('Eight selected submissions; recipe/seed proxies differ; v8 in-sample fixed-split parity excluded',fontsize=11)
fig.savefig(out/'validation_vs_leaderboard.png',dpi=180)
fig.savefig(out/'validation_vs_leaderboard.svg')
plt.close(fig)
