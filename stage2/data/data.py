from torch.utils.data import DataLoader
from .dataset import Stage2Dataset,collate
def get_data(manifest,batch_size,num_workers,shuffle):
    ds=Stage2Dataset(manifest);return ds,DataLoader(ds,batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True,persistent_workers=num_workers>0,collate_fn=collate)
