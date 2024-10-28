import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MiniDataset(Dataset):
    def __init__(self, dataloader, mini_dataset_ids, transform=None):
        self.dataloader = dataloader
        self.mini_dataset_ids = mini_dataset_ids
        self.transform = transform

        self.datas = []
        self.labels = []
        self.ids = []
        self._load_dataset()

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
        image = self._load_image(idx)
        label = self.labels[idx]
        id_value = self.ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, id_value

    def _load_image(self, idx):
        return Image.open(self.datas[idx])
