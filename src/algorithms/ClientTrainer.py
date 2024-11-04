import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from src.models.ClassfierModel import *
from src.models.ModelNetResnet18 import *
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, head, train_dataloader, test_dataloader, local_round_num=100, learning_rate=0.01):
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.local_round_num = local_round_num
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelNetResNet18()
        self.head = ClassifierModel()
        self.load_head(head)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.client_train_acc_rates = []
        self.client_test_acc_rates = []

    def print_info(self):
        print(f'    Client ID: {self.client_id}')
        print(f'    Client Device: {self.device}')
        print(f'    Client Model: ResNet18')
        print(f'    Client Local Round Num: {self.local_round_num}')
        print(f'    Client Dataset: ModelNet10 view {self.client_id}')
        print(f'    Client Data Batch Size: {self.train_dataloader.batch_size}')
        print(f'    Client Learning Rate: {self.learning_rate}')

    def load_head(self, head):
        new_state_dict = head.state_dict()
        load_state_dict = self.head.state_dict()
        for key in new_state_dict.keys():
            load_state_dict[key].data = new_state_dict[key].data.clone()
        self.head.load_state_dict(load_state_dict)
        self.head.to(self.device)

    def generate_client_embedding(self, mini_dataset_batch_size, mini_dataset_ids):
        """
        Generate client embedding
        :param mini_dataset_batch_size:
        :param mini_dataset_ids: 数据集id列表
        :return: {'id': 'embedding'}
        """
        print(f'    Client {self.client_id} is generating embeddings...')
        client_train_embeddings = {}
        client_test_embeddings = {}

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, mini_dataset_batch_size, mini_dataset_ids)
        self.model.eval()
        with torch.no_grad():
            for mini_batch in mini_dataloader:
                data, _, ids = mini_batch
                data = data.to(self.device)
                mini_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_train_embeddings[id_value] = mini_batch_embedding[i]
            for test_batch in self.test_dataloader:
                data, _, ids = test_batch
                data = data.to(self.device)
                test_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_test_embeddings[id_value] = test_batch_embedding[i]
        del mini_dataloader
        return client_train_embeddings, client_test_embeddings

    def train(self, head, client_train_embeddings, client_test_embeddings, mini_dataset_batch_size, mini_dataset_ids):
        print(f'    Client {self.client_id} is training...')
        self.load_head(head)

        mini_dataloader = generate_mini_dataloader(self.train_dataloader, mini_dataset_batch_size, mini_dataset_ids)

        self.head.eval()
        self.model.train()
        train_total = 0
        train_correct = 0
        for epoch in range(self.local_round_num):
            for i, data in enumerate(mini_dataloader, 0):
                inputs, labels, ids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                head_inputs = None
                for client_id, client_embedding in client_train_embeddings.items():
                    if client_id == self.client_id:
                        if head_inputs is None:
                            head_inputs = outputs
                        else:
                            head_inputs = torch.cat((head_inputs, outputs), dim=1)
                    else:
                        if head_inputs is None:
                            head_inputs = torch.stack([client_embedding[id_value] for id_value in ids], dim=0)
                        else:
                            head_inputs = torch.cat(
                                (head_inputs, torch.stack([client_embedding[id_value] for id_value in ids], dim=0)),
                                dim=1)
                head_inputs = head_inputs.to(self.device)
                head_outputs = self.head(head_inputs)
                _, predicted = torch.max(head_outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                self.optimizer.zero_grad()
                loss = self.criterion(head_outputs, labels)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        # print(f'    Client {self.client_id} accuracy on train set: {(100 * train_correct / train_total):.2f}%')
        self.client_train_acc_rates.append(100 * train_correct / train_total)

        self.head.eval()
        self.model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader, 0):
                inputs, labels, ids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                head_inputs = None
                for client_id, client_embedding in client_test_embeddings.items():
                    if client_id == self.client_id:
                        if head_inputs is None:
                            head_inputs = outputs
                        else:
                            head_inputs = torch.cat((head_inputs, outputs), dim=1)
                    else:
                        if head_inputs is None:
                            head_inputs = torch.stack([client_embedding[id_value] for id_value in ids], dim=0)
                        else:
                            head_inputs = torch.cat(
                                (head_inputs, torch.stack([client_embedding[id_value] for id_value in ids], dim=0)),
                                dim=1)
                head_inputs = head_inputs.to(self.device)
                head_outputs = self.head(head_inputs)
                _, predicted = torch.max(head_outputs, 1)
                # print(predicted)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        print(f'    Client {self.client_id} accuracy on test set: {(100 * test_correct / test_total):.2f}%')
        self.client_test_acc_rates.append(100 * test_correct / test_total)

        del mini_dataloader

    def test(self):
        print(f'    Client {self.client_id} is testing...')
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader, 0):
                inputs, labels, ids = data
                print(inputs.shape)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                print(outputs)


