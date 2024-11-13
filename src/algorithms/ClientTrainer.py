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
from src.datasets.DataloaderGenerator import *


class ClientTrainer:
    def __init__(self, client_id, head, train_dataset, test_dataset, client_dataset_batch_size, local_round_num=10, learning_rate=0.001, color=True):
        self.client_id = client_id
        self.client_dataset_batch_size = client_dataset_batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.client_dataset_batch_size, shuffle=False)
        self.local_round_num = local_round_num
        self.learning_rate = learning_rate
        self.weight_decay = 0.001
        self.color = color

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = ResNetForMNIST(self.color, 64)
        self.model = CNNForCifar(self.color, 64)
        self.head = ClassifierModel()
        self.load_head(head)
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
        print(f'    Client Model: CNN')
        print(f'    Client Local Round Num: {self.local_round_num}')
        print(f'    Client Dataset Color: {self.color}')
        print(f'    Client Data Batch Size: {self.client_dataset_batch_size}')
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

        mini_dataloader = generate_mini_dataloader(self.train_dataset, mini_dataset_batch_size, mini_dataset_ids)
        self.model.eval()
        with torch.no_grad():
            for mini_batch in mini_dataloader:
                data, _, ids = mini_batch
                data = data.to(self.device)
                mini_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_train_embeddings[id_value.item()] = mini_batch_embedding[i]
            for test_batch in self.test_dataloader:
                data, _, ids = test_batch
                data = data.to(self.device)
                test_batch_embedding = self.model(data)
                for i, id_value in enumerate(ids):
                    client_test_embeddings[id_value.item()] = test_batch_embedding[i]
        del mini_dataloader
        return client_train_embeddings, client_test_embeddings

    def train(self, head, client_train_embeddings, client_test_embeddings, mini_dataset_batch_size, mini_dataset_ids):
        print(f'    Client {self.client_id} is training...')
        self.load_head(head)

        mini_dataloader = generate_mini_dataloader(self.train_dataset, mini_dataset_batch_size, mini_dataset_ids)

        self.head.eval()
        self.model.train()
        epoch_train_loss_list = []
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
                            head_inputs = torch.stack([client_embedding[id_value.item()] for id_value in ids], dim=0)
                        else:
                            head_inputs = torch.cat((head_inputs, torch.stack([client_embedding[id_value.item()] for id_value in ids], dim=0)), dim=1)
                head_inputs = head_inputs.to(self.device)
                head_outputs = self.head(head_inputs)

                self.optimizer.zero_grad()
                loss = self.criterion(head_outputs, labels)
                l2_loss = self.model.l2_regularization_loss()
                loss += self.weight_decay * l2_loss
                loss.backward()
                epoch_train_loss_list.append(loss.item())
                self.optimizer.step()
            # self.scheduler.step()
        # print(f'    Client {self.client_id} train loss: {epoch_train_loss_list}')
        print(f'    Client {self.client_id} train loss avg: {sum(epoch_train_loss_list) / len(epoch_train_loss_list)}')
        self.client_test_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))

        self.head.eval()
        self.model.eval()
        test_correct = 0
        test_total = 0
        epoch_test_loss_list = []
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
                            head_inputs = torch.stack([client_embedding[id_value.item()] for id_value in ids], dim=0)
                        else:
                            head_inputs = torch.cat((head_inputs, torch.stack([client_embedding[id_value.item()] for id_value in ids], dim=0)), dim=1)
                head_inputs = head_inputs.to(self.device)
                head_outputs = self.head(head_inputs)
                _, predicted = torch.max(head_outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                loss = self.criterion(head_outputs, labels)
                l2_loss = self.model.l2_regularization_loss()
                loss += self.weight_decay * l2_loss
                epoch_test_loss_list.append(loss.item())

        # print(f'    Client {self.client_id} train loss: {epoch_test_loss_list}')
        print(f'    Client {self.client_id} test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        print(f'    Client {self.client_id} history accuracy on test set: {self.client_test_acc_rate_list}')
        print(f'    Client {self.client_id} accuracy on test set: {(100 * test_correct / test_total):.2f}%')
        self.client_test_acc_rate_list.append(100 * test_correct / test_total)
        self.client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

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


