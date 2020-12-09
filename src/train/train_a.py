import torch
import os
from torch.utils.data import Dataset, DataLoader
from src.train.train_base import *
import random


def get_data_loader(config, _type, shuffle=True, workers=10, drop_last=True, small=None):

    bs = config.train.batch_size if _type == 'train' else 1
    target = os.path.join(config.target_path, "{}.pkl".format(_type))
    dataset = ADataset(target) if small is None else ADataset(target)[:small]
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                             num_workers=workers, drop_last=drop_last)
    return data_loader


class ADataset(Dataset):
    def __init__(self, file):
        with open(file, 'rb') as f:
            self.data = torch.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
