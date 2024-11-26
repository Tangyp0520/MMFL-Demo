import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.MultiModelForCifar import *
from src.algorithms.SiloTrainer import *


class MMFL(object):
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path
        self.batch_size = 128
        self.train_dataset, self.test_dataset = generate_dataset('Multiple', dataset_root_path)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_idx = []
        self.split_train_dataset_index()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = MultiModelForCifar(self.device)
        self.global_model.to(self.device)
        self.global_round_num = 1
        self.global_learning_rate = 0.001
        self.global_weight_decay = 0.001

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        classifier_params = self.global_model.classifier.parameters()
        color_params = self.global_model.color_model.parameters()
        gray_params = self.global_model.gray_model.parameters()
        self.optimizer = optim.Adam([
            {'params': classifier_params, 'weight_decay': self.global_weight_decay},
            {'params': color_params, 'weight_decay': self.global_weight_decay, 'lr': 0},
            {'params': gray_params, 'weight_decay': self.global_weight_decay, 'lr': 0}
        ], lr=self.global_learning_rate, weight_decay=self.global_weight_decay)

        self.silo_num = 2
        self.silo_trainers = {}
        self.silo_ids = []
        for i in range(self.silo_num):
            self.silo_trainers[i] = SiloTrainer(i, self.train_dataset, self.test_dataset, self.batch_size)
            self.silo_ids.append(i)

        self.test_loss_list = []
        self.test_acc_rate_list = []
        self.print_info()

    def split_train_dataset_index(self):
        all_idx = list(range(len(self.train_dataset)))
        random.shuffle(all_idx)
        part_size = len(all_idx) // 5
        remainder = len(all_idx) % 5
        start = 0

        for i in range(5):
            end = start + part_size + (1 if i < remainder else 0)
            self.train_idx.append(all_idx[start:end])
            start = end

    def print_info(self):
        print(f'MMFL Device: {self.device}')
        print(f'MMFL Silo Num: {self.silo_num}')
        print(f'MMFL Dataset: CIFAR100')
        for _, silo_trainer in self.silo_trainers.items():
            silo_trainer.print_info()

    def model_aggregate(self, classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator):
        print(f'    MMFL Model Aggregate...')
        for name, param in self.global_model.classifier.state_dict().items():
            update_per_layer = classifier_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.color_model.state_dict().items():
            update_per_layer = color_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

        for name, param in self.global_model.gray_model.state_dict().items():
            update_per_layer = gray_weight_accumulator[name] / self.silo_num
            if param.type() != update_per_layer.type():
                param.add_(update_per_layer.to(torch.int64))
            else:
                param.add_(update_per_layer)

    def model_train(self, mini_train_idx):
        print(f'    MMFL Model Training...')
        train_dataloader = generate_mini_dataloader(self.train_dataset, self.batch_size, mini_train_idx)

        self.global_model.train()
        for _ in range(self.global_round_num):
            for batch in train_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.global_model(color, gray)

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

    def model_eval(self):
        print(f'    MMFL Model Evaluation...')
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
        print(f'    MMFL test loss avg: {sum(epoch_loss_list) / len(epoch_loss_list)}')
        print(f'    MMFL history accuracy on test set: {self.test_acc_rate_list}')
        print(f'    MMFL accuracy on test set: {(100 * correct / total):.2f}%')
        self.test_acc_rate_list.append(100 * correct / total)
        self.test_loss_list.append(sum(epoch_loss_list) / len(epoch_loss_list))

    def train(self, epoch):
        print(f'    MMFL train epoch: {epoch}')
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

        for silo_id, silo in self.silo_trainers.items():
            classifier_diff, color_diff, gray_diff = silo.train(self.global_model, mini_train_idx)
            for name, param in classifier_diff.items():
                classifier_weight_accumulator[name].add_(classifier_diff[name])
            for name, param in color_diff.items():
                color_weight_accumulator[name].add_(color_diff[name])
            for name, param in gray_diff.items():
                gray_weight_accumulator[name].add_(gray_diff[name])

        self.model_aggregate(classifier_weight_accumulator, color_weight_accumulator, gray_weight_accumulator)
        self.model_train(mini_train_idx)
        self.model_eval()
