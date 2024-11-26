import gc
import random

import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.ClientTrainer import *


class SiloTrainer(object):
    def __init__(self, silo_id, train_dataset, test_dataset, batch_size):
        self.silo_id = silo_id
        self.batch_size = batch_size
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        # self.train_idx = []
        # self.split_train_dataset_index()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silo_model = MultiModelForCifar(self.device)
        self.silo_model.to(self.device)
        self.silo_round_num = 1
        self.head_round_num = 1
        self.head_learning_rate = 0.001
        self.head_weight_decay = 0.001

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        classifier_params = self.silo_model.classifier.parameters()
        color_params = self.silo_model.color_model.parameters()
        gray_params = self.silo_model.gray_model.parameters()
        self.optimizer = optim.Adam([
            {'params': classifier_params, 'weight_decay': self.head_weight_decay},
            {'params': color_params, 'weight_decay': self.head_weight_decay, 'lr': 0},
            {'params': gray_params, 'weight_decay': self.head_weight_decay, 'lr': 0}
        ], lr=self.head_learning_rate, weight_decay=self.head_weight_decay)

        self.client_num = 4
        self.color_client_num = 2
        self.gray_client_num = 2
        self.client_trainers = {}
        self.client_ids = []
        for i, (dataset_type, color) in enumerate([('Cifar-gray', False), ('Cifar', True), ('Cifar-gray', False), ('Cifar', True)]):
            self.client_trainers[i] = ClientTrainer(i, self.train_dataset, self.test_dataset, self.batch_size, color=color)
            self.client_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []
        # self.print_info()

    # def split_train_dataset_index(self):
    #     all_idx = list(range(len(self.train_dataset)))
    #     random.shuffle(all_idx)
    #     part_size = len(all_idx) // 5
    #     remainder = len(all_idx) % 5
    #     start = 0
    #
    #     for i in range(5):
    #         end = start + part_size + (1 if i < remainder else 0)
    #         self.train_idx.append(all_idx[start:end])
    #         start = end

    def print_info(self):
        print(f'    Silo {self.silo_id} Model: ClassifierModel')
        print(f'    Silo {self.silo_id} Round Num: {self.head_round_num}')
        print(f'    Silo {self.silo_id} Dataset: CIFAR100')
        print(f'    Silo {self.silo_id} Dataset Batch Size: {self.batch_size}')
        print(f'    Silo {self.silo_id} Learning rate: {self.head_learning_rate}')
        print(f'    Silo {self.silo_id} Client Num: {self.client_num}')
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    Silo {self.silo_id} Model Aggregate...')
        for name, param in self.silo_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.silo_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name] / self.color_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.silo_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name] / self.gray_client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_train(self, mini_train_idx):
        print(f'    Silo {self.silo_id} Model Training...')
        train_dataloader = generate_mini_dataloader(self.train_dataset, self.batch_size, mini_train_idx)

        self.silo_model.train()
        for _ in range(self.head_round_num):
            for batch in train_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.silo_model(color, gray)

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

    def model_eval(self):
        print(f'    Silo {self.silo_id} Model Evaluation...')
        self.silo_model.eval()
        total = 0
        correct = 0
        epoch_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
                output = self.silo_model(color, gray)

                loss = self.criterion(output, labels)
                epoch_loss_list.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    Silo {self.silo_id} test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        # print(f'    Silo {self.silo_id} history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    Silo {self.silo_id} accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def silo_train(self, epoch, mini_train_idx):
        # print(f'    Silo {self.silo_id} train epoch: {epoch}')
        # mini_train_idx = self.train_idx[epoch % len(self.train_idx)]

        classifier_weight_accumulator = {}
        for name, param in self.silo_model.classifier.state_dict().items():
            classifier_weight_accumulator[name] = torch.zeros_like(param)
        color_weight_accumulator = {}
        for name, param in self.silo_model.color_model.state_dict().items():
            color_weight_accumulator[name] = torch.zeros_like(param)
        gray_weight_accumulator = {}
        for name, param in self.silo_model.gray_model.state_dict().items():
            gray_weight_accumulator[name] = torch.zeros_like(param)

        for client_id, client in self.client_trainers.items():
            classifier_diff, color_diff, gray_diff = client.train(self.silo_model, mini_train_idx)
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)
        self.model_train(mini_train_idx)
        # self.model_eval()

    def train(self, global_model, mini_train_idx):
        print(f'    Silo {self.silo_id} model fusion...')
        for name, param in global_model.state_dict().items():
            self.silo_model.state_dict()[name].copy_(param.clone())

        print(f'    Silo {self.silo_id} training...')
        for epoch in range(self.silo_round_num):
            self.silo_train(epoch, mini_train_idx)

        print(f'    Silo {self.silo_id} test...')
        self.model_eval()

        print(f'    Silo {self.silo_id} create diff dict...')
        classifier_diff = dict()
        for name, param in self.silo_model.classifier.state_dict().items():
            classifier_diff[name] = param - global_model.classifier.state_dict()[name]
        color_diff = dict()
        for name, param in self.silo_model.color_model.state_dict().items():
            color_diff[name] = param - global_model.color_model.state_dict()[name]
        gray_diff = dict()
        for name, param in self.silo_model.gray_model.state_dict().items():
            gray_diff[name] = param - global_model.gray_model.state_dict()[name]
        return classifier_diff, color_diff, gray_diff
