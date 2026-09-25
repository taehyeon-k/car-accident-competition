"""Bounded foreground experiment runner. Never touches existing runs."""
import concurrent.futures,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BATCHES={
 'replicate':[
  ['--name','LN_soft','--sigma','-1','--seed','1'],
  ['--name','LN_exact','--sigma','0','--seed','1'],
  ['--name','TCN_frame6','--sigma','6','--kind','tcn','--seed','1'],
  ['--name','LN_soft','--sigma','-1','--seed','2'],
  ['--name','LN_exact','--sigma','0','--seed','2'],
  ['--name','TCN_frame6','--sigma','6','--kind','tcn','--seed','2'],
 ],
 'first':[
  ['--name','LN_soft','--sigma','-1'],
  ['--name','LN_frame6','--sigma','6'],
  ['--name','LN_exact','--sigma','0'],
  ['--name','TCN_frame6','--sigma','6','--kind','tcn'],
 ],
 'second':[
  ['--name','LN_nexar_ft','--sigma','3','--nexar-only','--lr','.0002','--selection-radius','8','--init','stage2/nexar65_experiments/results/LN_soft/seed0/checkpoint.pt'],
  ['--name','LN_nexar_prior_ft','--sigma','3','--nexar-only','--prior','--lr','.0002','--selection-radius','8','--init','stage2/nexar65_experiments/results/LN_soft/seed0/checkpoint.pt'],
  ['--name','LN_repeat2','--sigma','3','--nexar-repeat','2','--selection-radius','8'],
  ['--name','LN_prior_repeat2','--sigma','3','--nexar-repeat','2','--prior','--selection-radius','8'],
 ],
}

def run(args):
    name=args[args.index('--name')+1];seed=args[args.index('--seed')+1] if '--seed' in args else '0'
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'{name}_seed{seed}.log'
    with path.open('x') as f:
        p=subprocess.run([sys.executable,'-m','stage2.nexar65_experiments.train',*args],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT.parents[1])
    print('FINISHED',name,seed,p.returncode,flush=True)
    return p.returncode

if __name__=='__main__':
    jobs=BATCHES[sys.argv[1]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: codes=list(ex.map(run,jobs))
    sys.exit(any(codes))
