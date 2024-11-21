import gc

import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.ClientTrainer import *


class MMFl(object):
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.head_round_num = 10
        self.dataset_size = 128
        self.train_dataset, self.test_dataset = generate_dataset('Multiple', dataset_root_path)
        self.train_idx = []
        self.split_train_dataset_index()
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.dataset_size, shuffle=True)

        self.global_model = MultiModelForCifar(self.device)
        self.global_model.to(self.device)
        # self.head_learn_rate = 0.001
        # self.weight_decay = 0.001
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.optimizer = torch.optim.Adam(self.head.parameters(), lr=self.head_learn_rate, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.global_protos = []

        self.client_num = 2
        self.client_batch_size = 128
        self.client_trainers = {}
        self.client_ids = []
        for i, (dataset_type, color) in enumerate([('Cifar-gray', False), ('Cifar', True)]):
            self.client_trainers[i] = ClientTrainer(i, self.train_dataset, self.test_dataset, self.client_batch_size, color=color)
            self.client_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []
        self.print_info()
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()
            # client_trainer.test()

    def split_train_dataset_index(self):
        all_idx = range(len(self.train_dataset))
        part_size = len(all_idx) // 5
        remainder = len(all_idx) % 5
        start = 0

        for i in range(5):
            end = start + part_size + (1 if i < remainder else 0)
            self.train_idx.append(all_idx[start:end])
            start = end

    def print_info(self):
        print(f'MMFL device: {self.device}')
        print(f'Head Model: ClassifierModel')
        # print(f'Head Round Num: {self.head_round_num}')
        print(f'Head Dataset: CIFAR')
        # print(f'Head Dataset Batch Size: {self.head_dataset_batch_size}')
        # print(f'Head Learning rate: {self.head_learn_rate}')
        print(f'Client Num: {self.client_num}')
        print(f'Client Dataset Batch Size: {self.client_batch_size}')
        # print(f'Mini Dataset Size: {self.mini_dataset_size}')
        # print(f'Mini Dataset Batch Size: {self.mini_dataset_batch_size}')

    def proto_aggregate(self, local_protos_lists):
        global_protos = []
        for i in range(len(local_protos_lists[0])):
            class_protos = torch.stack([local_protos[i] for local_protos_lists in local_protos_lists for local_protos in local_protos_lists])
            global_proto = torch.mean(class_protos, dim=0)
            global_protos.append(global_proto)
        return torch.stack(global_protos)

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    Model Aggregate...')
        for name, param in self.global_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.client_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name]
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name]
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_eval(self):
        print(f'    Server Model Evaluation...')
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
        print(f'    Server test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        print(f'    Server history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    Server accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def train(self, epoch):
        print(f'Global train epoch: {epoch}')
        mini_train_idx = self.train_idx[epoch % len(self.train_idx)]
        train_dataloader = generate_mini_dataloader(self.train_dataset, self.dataset_size, mini_train_idx)

        classifier_weight_accumulator = {}
        for name, param in self.global_model.classifier.state_dict().items():
            classifier_weight_accumulator[name] = torch.zeros_like(param)
        color_weight_accumulator = {}
        for name, param in self.global_model.color_model.state_dict().items():
            color_weight_accumulator[name] = torch.zeros_like(param)
        gray_weight_accumulator = {}
        for name, param in self.global_model.gray_model.state_dict().items():
            gray_weight_accumulator[name] = torch.zeros_like(param)

        local_protos_lists = []
        for client_id, client in self.client_trainers.items():
            classifier_diff, color_diff, gray_diff = client.train(self.global_model, self.global_protos, mini_train_idx)
            local_protos = client.generate_proto()
            local_protos_lists.append(local_protos)
            # for name, param in self.global_model.state_dict().items():
            #     weight_accumulator[name].add_(diff[name])
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)
        self.global_protos = self.proto_aggregate(local_protos_lists)
        self.model_eval()
