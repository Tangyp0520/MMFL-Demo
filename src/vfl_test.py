import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import gc
import random
import datetime

import torch
import torch.nn as nn

from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.ClientTrainer import *
from src.utils.ExcelUtil import *


class VFLTrainer(object):
    def __init__(self, dataset_root_path):
        self.batch_size = 128
        self.train_dataset, self.test_dataset = generate_dataset('Multiple', dataset_root_path)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_idx = []
        self.split_train_dataset_index()

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.global_model = MultiModelForCifar(self.device)
        self.global_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.client_num = 2
        self.color_client_num = 1
        self.gray_client_num = 1
        self.client_trainers = {}
        self.client_ids = []
        for i, (dataset_type, color) in enumerate([('Cifar', 1), ('Cifar-gray', 2)]):
            self.client_trainers[i] = ClientTrainer(i, self.train_dataset, self.test_dataset, self.batch_size, color=color)
            self.client_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []
        self.print_info()

    def split_train_dataset_index(self):
        all_idx = list(range(len(self.train_dataset)))
        part_num = 10
        random.shuffle(all_idx)
        part_size = len(all_idx) // part_num
        remainder = len(all_idx) % part_num
        start = 0

        for i in range(part_num):
            end = start + part_size + (1 if i < remainder else 0)
            self.train_idx.append(all_idx[start:end])
            start = end

    def print_info(self):
        print(f'    VFL Model: ClassifierModel')
        print(f'    VFL Dataset: CIFAR100')
        print(f'    VFL Client Num: {self.client_num}')
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    VFL Model Aggregate...')
        for name, param in self.global_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name] / self.color_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name] / self.gray_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_train(self):
        self.global_model.train()
        for batch in self.test_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
            self.global_model(color, gray)

    def model_eval(self):
        print(f'    VFL Model Evaluation...')
        self.model_train()
        self.global_model.eval()
        total = 0
        correct = 0
        epoch_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
                output = self.global_model(color, gray)

                loss = self.criterion(output, labels)
                epoch_loss_list.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    HFL test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        print(f'    HFL history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    HFL accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def train(self, epoch):
        print(f'    VFL train epoch: {epoch}')
        mini_train_idx = self.train_idx[epoch % len(self.train_idx)]

        classifier_weight_accumulator = {}
        for name, param in self.global_model.classifier.state_dict().items():
            classifier_weight_accumulator[name] = torch.zeros_like(param)
        color_weight_accumulator = {}
        for name, param in self.global_model.color_model.state_dict().items():
            color_weight_accumulator[name] = torch.zeros_like(param)
        gray_weight_accumulator = {}
        for name, param in self.global_model.gray_model.state_dict().items():
            gray_weight_accumulator[name] = torch.zeros_like(param)

        for _, client in self.client_trainers.items():
            classifier_diff, color_diff, gray_diff = client.train(self.global_model, mini_train_idx)
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)
        self.model_eval()


if __name__ == '__main__':
    print("This is VFL Demo")
    # dataset_root_path = 'D:\\.download\\ModelNet10\\dataset'
    # dataset_root_path = '/home/data2/duwenfeng/datasets/ModelNet10'
    # dataset_root_path = 'D:\.download\MNIST-M\data\mnist_m'
    dataset_root_path = '/home/data2/duwenfeng/datasets/MNIST'
    global_round_num = 50

    print('VFL train start...')
    print(f'Global Round Num: {global_round_num}')
    vfl = VFLTrainer(dataset_root_path)
    for epoch in range(global_round_num):
        vfl.train(epoch)
    print('VFL train end...')

    print('Save result...')
    file_head_name = '_vfl_resnet_cifar100_'
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')

    head_test_loss_lists = vfl.test_loss_list
    client_test_loss_lists = {}
    excel_file_name = (date_str + file_head_name + 'loss')
    save_acc_to_excel(excel_file_name, head_test_loss_lists, {})

    head_test_acc_rates = vfl.test_acc_rate_list
    excel_file_name = (date_str + file_head_name + 'acc')
    save_acc_to_excel(excel_file_name, head_test_acc_rates, {})
