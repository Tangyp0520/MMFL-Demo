import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MiniDataset(Dataset):
    def __init__(self, dataloader, mini_dataset_ids):
        self.dataloader = dataloader
        self.mini_dataset_ids = mini_dataset_ids

        self.datas = []
        self.labels = []
        self.ids = []
        self._load_dataset()
        print(self.datas)
        print(self.labels)
        print(self.ids)
        print(len(self.datas))
        print(len(self.labels) == len(self.ids) and len(self.datas) == len(self.labels))

    def _load_dataset(self):
        for batch in self.dataloader:
            batch_data, batch_labels, batch_ids = batch
            for data, label, id_value in zip(batch_data, batch_labels, batch_ids):
                if id_value in self.mini_dataset_ids:
                    self.datas.append(data)
                    self.labels.append(label)
                    self.ids.append(id_value)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx], self.ids[idx]
