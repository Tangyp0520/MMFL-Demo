import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class ModelNetDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.ids = []
        self._load_dataset()

    def _load_dataset(self):
        for folder_name in os.listdir(self.root_dir):
            label = folder_name
            if self.train:
                folder_path = os.path.join(self.root_dir, 'train', folder_name)
            else:
                folder_path = os.path.join(self.root_dir, 'test', folder_name)

            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    self.image_paths.append(file_path)
                    self.labels.append(label)
                    self.ids.append(file_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self._load_image(idx)
        label = self.labels[idx]
        id_value = self.ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, id_value

    def _load_image(self, idx):
        return Image.open(self.image_paths[idx])