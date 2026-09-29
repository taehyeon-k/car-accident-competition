"""Isolated target-transport arm; same CPU trainer/model/optimizer as baseline."""
from . import train_cpu
from .transport_targets import install
import json
import sys
from pathlib import Path
import torch

if __name__=='__main__':
    install(train_cpu)
    train_cpu.main()
    folder=Path(sys.argv[sys.argv.index('--output')+1])
    policy='native_index_gaussian_mass_nearest_augmented_frame'
    config=json.loads((folder/'config.json').read_text());config['target_policy']=policy
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    checkpoint['config']['target_policy']=policy
    torch.save(checkpoint,folder/'checkpoint.pt')
