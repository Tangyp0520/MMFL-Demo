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
    def __init__(self, client_id, train_dataset, test_dataset, client_batch_size, local_round_num=1,
                 learning_rate=0.001, color=True):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_batch_size = client_batch_size
        self.class_num = 10

        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.client_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.client_batch_size, shuffle=False)

        self.local_round_num = local_round_num
        self.learning_rate = learning_rate
        self.weight_decay = 0.001
        self.proto_reg = 0.1
        self.color = color

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiModelForCifar(self.device)
        self.model.to(self.device)

        classifier_params = self.model.classifier.parameters()
        color_params = self.model.color_model.parameters()
        gray_params = self.model.gray_model.parameters()

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.color:
            self.optimizer = optim.Adam([
                {'params': classifier_params, 'weight_decay': self.weight_decay},
                {'params': color_params, 'weight_decay': self.weight_decay},
                {'params': gray_params, 'weight_decay': self.weight_decay, 'lr': 0}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam([
                {'params': classifier_params, 'weight_decay': self.weight_decay},
                {'params': color_params, 'weight_decay': self.weight_decay, 'lr': 0},
                {'params': gray_params, 'weight_decay': self.weight_decay}
            ], lr=self.learning_rate, weight_decay=self.weight_decay)
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

    def generate_proto(self):
        self.model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_dataset):
                self.optimizer.zero_grad()
                color, gray, label, _ = batch
                color, gray = color.to(self.device), gray.to(self.device)
                color_embedding = self.model.color_model(color)
                gray_embedding = self.model.gray_model(gray)
                embedding = torch.cat((color_embedding, gray_embedding), dim=1)
                embeddings.append(embedding)
                labels.append(label)
        # self.model.train()

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels)
        protos = []
        for i in range(self.class_num):
            class_embeddings = embeddings[labels == i]
            prototype = torch.mean(class_embeddings, dim=0)
            protos.append(prototype)
        return torch.stack(protos)

    def generate_local_proto(self, train_dataset, global_protos):
        self.model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataset):
                self.optimizer.zero_grad()
                color, gray, label, _ = batch
                color, gray = color.to(self.device), gray.to(self.device)
                color_embedding = self.model.color_model(color)
                gray_embedding = self.model.gray_model(gray)
                embedding = torch.cat((color_embedding, gray_embedding), dim=1)
                embeddings.append(embedding)
                labels.append(label)
        self.model.train()

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels)
        protos = []
        for i in range(self.class_num):
            if i in labels:
                class_embeddings = embeddings[labels == i]
                prototype = torch.mean(class_embeddings, dim=0)
                protos.append(prototype)
            else:
                protos.append(global_protos[i])
        return torch.stack(protos)

    def compute_proto_loss(self, train_dataloader, global_protos):
        local_protos = self.generate_local_proto(train_dataloader.dataset, global_protos)
        proto_loss = 0
        for i in range(len(global_protos)):
            proto_loss += torch.norm(local_protos[i] - global_protos[i])
        return proto_loss

    def train(self, global_model, global_protos, mini_train_idx):
        # 小批量数据集生成
        train_dataloader = generate_mini_dataloader(self.train_dataset, self.client_batch_size, mini_train_idx)
        # 模型聚合
        print(f'    Client {self.client_id} model fusion...')
        for name, param in global_model.state_dict().items():
            self.model.state_dict()[name].copy_(param.clone())

        print(f'    Client {self.client_id} train...')
        self.model.train()
        epoch_train_loss_list = []
        for epoch in range(self.local_round_num):
            for batch in train_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(color, gray)
                loss = self.criterion(output, labels)
                proto_loss = self.compute_proto_loss(train_dataloader, global_protos)
                loss += self.proto_reg * proto_loss
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
                color, gray, labels, _ = batch
                color, gray, labels = color.to(self.device), gray.to(self.device), labels.to(self.device)
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
        classifier_diff = dict()
        for name, param in self.model.classifier.state_dict().items():
            classifier_diff[name] = param - global_model.classifier.state_dict()[name]
        color_diff = dict()
        for name, param in self.model.color_model.state_dict().items():
            color_diff[name] = param - global_model.color_model.state_dict()[name]
        gray_diff = dict()
        for name, param in self.model.gray_model.state_dict().items():
            gray_diff[name] = param - global_model.gray_model.state_dict()[name]
        return classifier_diff, color_diff, gray_diff
