import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from src.models.ClassfierModel import *
from src.models.Resnet18ForModelNet import *
from src.models.ResNetForMNIST import *
from src.models.CNNForCifar import *
from src.models.MultiModelForCifar import *
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, train_dataset, test_dataset, client_batch_size, local_round_num=10, learning_rate=0.001, color=True):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_batch_size = client_batch_size

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.client_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.client_batch_size, shuffle=False)

        self.local_round_num = local_round_num
        self.learning_rate = learning_rate
        self.weight_decay = 0.001
        self.color = color

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiModelForCifar()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        # self.client_train_acc_rates = []
        self.client_train_loss_list = []
        self.client_test_loss_list = []
        self.client_test_acc_rate_list = []

    def print_info(self):
        print(f'    Client ID: {self.client_id}')
        print(f'    Client Device: {self.device}')
        print(f'    Client Model: MultiModelForCifar')
        print(f'    Client Local Round Num: {self.local_round_num}')
        print(f'    Client Dataset Color: {self.color}')
        print(f'    Client Data Batch Size: {self.client_batch_size}')
        print(f'    Client Learning Rate: {self.learning_rate}')

    def train(self, global_model):
        # 模型聚合
        print(f'    Client {self.client_id} model fusion...')
        for name, param in global_model.state_dict().items():
            self.model.state_dict()[name].copy_(param.clone())

        print(f'    Client {self.client_id} train...')
        self.model.train()
        epoch_train_loss_list = []
        for epoch in range(self.local_round_num):
            for batch in self.train_dataloader:
                color = None
                gray = None
                if self.color:
                    color, _, labels, _ = batch
                    color, labels = color.to(self.device), labels.to(self.device)
                else:
                    _, gray, labels, _ = batch
                    gray, labels = gray.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(color, gray)
                loss = self.criterion(output, labels)
                loss.backward()
                epoch_train_loss_list.append(loss.item())
                self.optimizer.step()
        print(f'    Client {self.client_id} train loss avg: {sum(epoch_train_loss_list) / len(epoch_train_loss_list)}')
        self.client_train_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))

        print(f'    Client {self.client_id} test...')
        self.model.eval()
        total = 0
        correct = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                color = None
                gray = None
                if self.color:
                    color, _, labels, _ = batch
                    color, labels = color.to(self.device), labels.to(self.device)
                else:
                    _, gray, labels, _ = batch
                    gray, labels = gray.to(self.device), labels.to(self.device)
                output = self.model(color, gray)
                loss = self.criterion(output, labels)
                epoch_test_loss_list.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    Client {self.client_id} test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        print(f'    Client {self.client_id} history accuracy on test set: {self.client_test_acc_rate_list}')
        print(f'    Client {self.client_id} accuracy on test set: {(100 * correct / total):.2f}%')
        self.client_test_acc_rate_list.append(100 * correct / total)
        self.client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

        print(f'    Client {self.client_id} create diff dict...')
        diff = dict()
        for name, param in self.model.state_dict().items():
            diff[name] = param - global_model.state_dict()[name]
        return diff

