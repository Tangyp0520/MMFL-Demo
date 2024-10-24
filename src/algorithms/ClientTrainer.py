import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from src.models.ModelNetResnet18 import *
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, head, train_dataloader, test_dataloader, local_round_num=100, learning_rate=0.01):
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.local_round_num = local_round_num
        self.learning_rate = learning_rate

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ModelNetResNet18()
        self.head = copy.deepcopy(head)
        self.model.to(self.device)
        self.head.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.client_acc_rates = []

    def print_info(self):
        print(f'    Client ID: {self.client_id}')
        print(f'    Client Device: {self.device}')
        print(f'    Client Model: ResNet18')
        print(f'    Client Local Round Num: {self.local_round_num}')
        print(f'    Client Dataset: ModelNet10 view {self.client_id}')
        print(f'    Client Data Batch Size: {self.train_dataloader.batch_size}')
        print(f'    Client Learning Rate: {self.learning_rate}')

    def generate_client_embedding(self, mini_dataset_batch_size, mini_dataset_ids):
        """
        Generate client embedding
        :param mini_dataset_batch_size:
        :param mini_dataset_ids: 数据集id列表
        :return: {'id': 'embedding'}
        """
        client_train_embeddings = {}
        client_test_embeddings = {}

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, mini_dataset_batch_size, mini_dataset_ids)
        self.model.eval()
        with torch.no_grad():
            for mini_batch in mini_dataloader:
                data, labels, ids = mini_batch
                data = data.to(self.device)
                mini_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_train_embeddings[id_value] = mini_batch_embedding[i]
            for test_batch in self.test_dataloader:
                data, labels, ids = test_batch
                data = data.to(self.device)
                test_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_test_embeddings[id_value] = test_batch_embedding[i]
        return client_train_embeddings, client_test_embeddings

    def train(self, head, client_train_embeddings, mini_dataset_batch_size, mini_dataset_ids):
        print(f'    Client {self.client_id} is training...')
        self.head = copy.deepcopy(head)
        self.head.eval()
        self.model.train()

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, mini_dataset_batch_size, mini_dataset_ids)

        for epoch in range(self.local_round_num):
            for i, data in enumerate(mini_dataloader, 0):
                inputs, labels, ids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                head_outputs = self.head(outputs)
                loss = self.criterion(head_outputs, labels)
                # 根据样本id获取其余客户端对应embedding 根据此embedding获得损失加上本客户端新生embedding并平均 将平均损失作为整体损失反向传播
                total_loss = loss.item()
                for client_id, client_embedding in client_train_embeddings.items():
                    if client_id != self.client_id:
                        data = [client_embedding[id_value] for id_value in ids]
                        data = torch.tensor(data).to(self.device)
                        this_outputs = self.head(data)
                        this_loss = self.criterion(this_outputs, labels)
                        total_loss += this_loss.item()
                total_loss /= len(client_train_embeddings)
                torch.tensor(total_loss).backward()
                self.optimizer.step()
        self.scheduler.step()

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(mini_dataloader, 0):
                inputs, labels, ids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                head_outputs = self.head(outputs)
                _, predicted = torch.max(head_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Client {self.client_id} accuracy on test set: {(100 * correct / total):.2f}%')
        self.client_acc_rates.append(100 * correct / total)
