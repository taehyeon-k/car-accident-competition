"""Summarize controlled ablations without touching the submission package."""
import csv
import json
from pathlib import Path
import numpy as np
from stage2.geometry_pretrain.ablate_normalized_probe import OUT, ROOT, VARIANTS, score


def fmt(values):
    return f'{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f}'


def main():
    results=json.loads((OUT/'results.json').read_text())
    assert len(results)==18,'Wait until all six variants and three seeds finish'
    groups={cfg['name']:[r for r in results if r['config']['name']==cfg['name']] for cfg in VARIANTS}
    names={'baseline128':'Tuned, 128 frames + jitter + TCN','frames64':'64 frames','frames256':'256 frames',
           'no_jitter':'No sampling jitter','no_temporal':'No temporal convolutions','original_backbone':'Original untuned backbone'}
    base=groups['baseline128']
    summary=[]
    for name,runs in groups.items():
        summary.append(dict(variant=name,best_mean=float(np.mean([r['best']['score'] for r in runs])),
                            best_std=float(np.std([r['best']['score'] for r in runs],ddof=1)),
                            final_mean=float(np.mean([r['final']['score'] for r in runs])),
                            final_std=float(np.std([r['final']['score'] for r in runs],ddof=1)),
                            stride2_mean=float(np.mean([r['stress']['2']['score'] for r in runs])),
                            stride4_mean=float(np.mean([r['stress']['4']['score'] for r in runs])),
                            mean_seconds=float(np.mean([r['seconds'] for r in runs]))))
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    best=max(summary,key=lambda r:r['best_mean']); fixed=max(summary,key=lambda r:r['final_mean'])
    lines=['# Stage 2 FPS-independent probe ablation study','',
           '## What was tested','',
           'Six configurations × three seeds (0, 1, 2): **18 fits**. Same original split: '
           '**201 training videos / 50 validation videos**, same 40-epoch budget, batch 8, '
           'AdamW (LR 0.001, weight decay 0.05), OneCycle schedule and competition-weighted losses. '
           'The backbone is frozen and precomputed features are reused. Only the small temporal head is trained. '
           'All configurations sample normalized clip positions and need no FPS at inference.', '',
           'Scores use native-frame ground-truth event times: ±0.3 s accuracy for entry/collision, '
           'and Macro-F1 for entry side/evasion. Composite = 0.35 entry + 0.35 collision + 0.15 side + 0.15 evasion. '
           'Source FPS is used to construct training losses/evaluation times, never as a prediction input.', '',
           '**Best validation** selects the best epoch independently per run. **Fixed epoch 40** '
           'avoids cherry-picking an epoch and reveals whether improvements persist. '
           'Values are mean ± sample standard deviation across three seeds; neither column is an untouched test score.', '',
           '## Results','', '| Configuration | Best validation | Fixed epoch 40 | Δ best vs baseline | Best epochs |',
           '|---|---:|---:|---:|---|']
    for item in summary:
        name=item['variant']; runs=groups[name]
        delta=np.mean([r['best']['score']-b['best']['score'] for r,b in zip(runs,base)])
        lines.append(f"| {names[name]} | {fmt([r['best']['score'] for r in runs])} | {fmt([r['final']['score'] for r in runs])} | {delta:+.4f} | {', '.join(str(r['best_epoch']) for r in runs)} |")
    lines+=['','## Component scores at each selected checkpoint','',
            '| Configuration | Entry accuracy | Collision accuracy | Side Macro-F1 | Evasion Macro-F1 |',
            '|---|---:|---:|---:|---:|']
    for name,runs in groups.items():
        cells=[f"{np.mean([r['best'][k] for r in runs]):.4f}" for k in ('entry_accuracy','collision_accuracy','side_f1','evasion_f1')]
        lines.append('| '+names[name]+' | '+' | '.join(cells)+' |')
    lines+=['','## Lower-frame-density stress test','',
            'Hold each selected checkpoint fixed and remove cached frames before normalized sampling. '
            'Stride 2/4 roughly halves/quarters the original ~10 Hz feature cadence. '
            'This tests missing visual evidence; changing FPS metadata alone has no effect. '
            'The last retained frame may shift slightly under subsampling, as it would with actual sparse input.', '',
            '| Configuration | Original density | Half density | Quarter density |','|---|---:|---:|---:|']
    for item in summary:
        lines.append(f"| {names[item['variant']]} | {item['best_mean']:.4f} | {item['stride2_mean']:.4f} | {item['stride4_mean']:.4f} |")
    lines+=['','## Interpretation and uncertainty','',
            f"Highest mean selected-checkpoint score: **{names[best['variant']]} ({best['best_mean']:.4f})**. "
            f"Highest mean fixed-epoch score: **{names[fixed['variant']]} ({fixed['final_mean']:.4f})**.", '',
            'Three seeds quantify optimization variability on one split, not generalization across new videos. '
            'There are only 50 validation clips; one additional correct event changes its accuracy by 0.02 '
            'and the composite by 0.007. Selecting among epochs and configurations introduces validation-selection optimism. '
            'Treat small deltas as tentative until video-level cross-validation or a fresh holdout confirms them.', '',
            '128 fixed positions remove dependence on an assumed FPS, but they do not guarantee equal accuracy '
            'for every possible duration or frame rate. Extremely long clips have coarser event localization, '
            'and low frame rates may omit the event entirely. Higher frame counts increase DINO encoding cost.', '',
            'All ablation fits use only the 201 training labels. The existing all-251 submission checkpoint '
            'is not used for held-out evaluation. The tuned backbone was trained by the preceding geometry experiment; '
            'this study does not retrain it. Feature-cache preprocessing and split provenance follow that experiment.', '',
            '## Artifacts and reproducibility','',
            f"Total fit/evaluation wall time (sum of runs): {sum(r['seconds'] for r in results)/60:.1f} minutes. "
            f"Peak allocated GPU memory among runs: {max(r['peak_gpu_MiB'] for r in results):.1f} MiB.", '',
            'Detailed curves, selected checkpoints and per-video predictions: '
            '`/workspace/outputs/geometry_pretrain/fps_ablation/<variant>/seed<seed>/`.', '',
            '```bash','python -m stage2.geometry_pretrain.ablate_normalized_probe',
            'python -m stage2.geometry_pretrain.report_normalized_ablation','```','',
            '**Submission files were not modified by this study.** Their SHA-256 manifest is recorded '
            'before and checked after the experiments. No ablation winner has been installed.', '']
    path=ROOT/'reports/stage2_fps_ablation.md'; path.write_text('\n'.join(lines))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(path); print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
