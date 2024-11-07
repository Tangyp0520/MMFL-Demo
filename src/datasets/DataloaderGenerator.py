import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.datasets.ModelNetDataset import *
from src.datasets.MiniDataset import *
from src.datasets.MNISTDataset import *
from src.datasets.MNISTMDataset import *


def generate_dataloader(dataset_type, batch_size, data_path=None, load_type=True):
    if dataset_type == 'MNIST':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # 加载 MNIST 训练集
        # train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        train_dataset = MNISTDataset(data_path+'/mnist', train=True, transform=transform, load_type=load_type)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载 MNIST 测试集
        # test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        test_dataset = MNISTDataset(data_path+'/mnist', train=False, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    elif dataset_type == 'MNIST-M':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        # 加载MNIST-M训练集
        # train_dataset = torchvision.datasets.MNISTM(root='../data', train=True, download=True, transform=transform)
        train_dataset = MNISTMDataset(data_path+'/mnist_m', train=True, transform=transform, load_type=load_type)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载MNIST-M测试集
        # test_dataset = torchvision.datasets.MNISTM(root='../data', train=False, download=True, transform=transform)
        test_dataset = MNISTMDataset(data_path+'/mnist_m', train=False, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    elif dataset_type == 'ModelNet10':
        # 定义数据预处理转换
        transform = transforms.Compose([
            transforms.Lambda(convert_four_to_three_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载ModelNet训练集
        train_dataset = ModelNetDataset(root_dir=data_path, train=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 加载ModelNet测试集
        test_dataset = ModelNetDataset(root_dir=data_path, train=False, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader


def generate_mini_dataloader(dataloader, mini_dataset_batch_size, mini_dataset_ids, color=True):
    """
    根据全体数据集dataloader及选择的id列表生成新的dataloader
    :param dataloader:
    :param mini_dataset_batch_size:
    :param mini_dataset_ids:
    :return:
    """
    # 定义数据预处理转换
    transform_color = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_bk = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if color:
        transform = transform_color
    else:
        transform = transform_bk
    mini_dataset = MiniDataset(dataloader, mini_dataset_ids, transform)
    mini_dataloader = DataLoader(mini_dataset, batch_size=mini_dataset_batch_size, shuffle=True)
    return mini_dataloader


def convert_four_to_three_channels(img):
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        return Image.fromarray(img_array[:, :, :3])
    else:
        return img
