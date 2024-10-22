import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from src.models.ModelNetResnet18 import *
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, head, train_dataloader, test_dataloader, local_epoch=100, learning_rate=0.01):
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.local_epoch = local_epoch
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelNetResNet18()
        self.head = copy.deepcopy(head)
        self.model.to(self.device)
        self.head.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def generate_client_embedding(self, mini_batch_ids):
        client_train_embeddings = {}
        client_test_embeddings = {}

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, 64, mini_batch_ids)
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

    def train(self, head, client_embeddings, mini_batch_ids):
        self.head = copy.deepcopy(head)
        self.head.eval()
        self.model.train()

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, 64, mini_batch_ids)

        for epoch in range(self.local_epoch):
            for i, data in enumerate(mini_dataloader, 0):
                inputs, labels, ids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                head_outputs = self.head(outputs)
                loss = self.criterion(head_outputs, labels)
                total_loss = loss.item()
                for client_id, client_embedding in client_embeddings.items():
                    if client_id != self.client_id:
                        data = [client_embedding[id_value] for id_value in ids]
                        data = torch.tensor(data).to(self.device)
                        this_outputs = self.head(data)
                        this_loss = self.criterion(this_outputs, labels)
                        total_loss += this_loss.item()
                total_loss /= len(client_embeddings)
                torch.tensor(total_loss).backward()
                self.optimizer.step()
        self.scheduler.step()

