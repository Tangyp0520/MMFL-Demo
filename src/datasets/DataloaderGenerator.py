import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ModelNetDataset import *
from MiniDataloader import *


def generate_dataloader(dataset_type, batch_size, data_path=None):
    if dataset_type == 'MNIST':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 加载 MNIST 训练集
        train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载 MNIST 测试集
        test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    elif dataset_type == 'MNIST-M':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # 加载MNIST-M训练集
        train_dataset = torchvision.datasets.MNISTM(root='../data', train=True, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载MNIST-M测试集
        test_dataset = torchvision.datasets.MNISTM(root='../data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    elif dataset_type == 'ModelNet10':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # 加载ModelNet训练集
        train_dataset = ModelNetDataset(root_dir=data_path, train=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载ModelNet测试集
        test_dataset = ModelNetDataset(root_dir=data_path, train=False, transform=transform)
        test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader


def generate_mini_dataloader(dataloader, batch_size, mini_batch_ids, train=True):
    mini_dataset = MiniDataset(dataloader, mini_batch_ids)
    if train:
        mini_dataloader = DataLoader(mini_dataset, batch_size=batch_size, shuffle=True)
    else:
        mini_dataloader = DataLoader(mini_dataset, batch_size=batch_size, shuffle=False)
    return mini_dataloader
