import gc

import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.algorithms.ClientTrainer import *


class MMFl(object):
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_round_num = 10
        self.head_dataset_batch_size = 128
        self.head_train_dataset, self.head_test_dataset = generate_dataset('Cifar', dataset_root_path)
        self.head_test_dataloader = DataLoader(dataset=self.head_test_dataset, batch_size=self.head_dataset_batch_size, shuffle=True)
        self.head_train_ids = []
        self.read_dataset_ids()

        self.head = ClassifierModel()
        self.head.to(self.device)
        self.head_learn_rate = 0.001
        self.weight_decay = 0.001
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=self.head_learn_rate, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.client_num = 2
        self.client_dataset_batch_size = 128
        self.client_trainers = {}
        self.client_ids = []
        for i, (dataset_type, color) in enumerate([('Cifar-gray', False), ('Cifar', True)]):
            train_dataset, test_dataset = generate_dataset(dataset_type, dataset_root_path)
            self.client_trainers[i] = ClientTrainer(i, self.head, train_dataset, train_dataset, self.client_dataset_batch_size, color=color)
            self.client_ids.append(i)

        self.mini_dataset_size = len(self.head_train_ids) // self.client_num
        self.mini_dataset_ids = []
        self.mini_dataset_batch_size = 128

        self.head_train_loss_list = []
        self.head_test_loss_list = []
        self.head_test_acc_rate_list = []
        self.print_info()
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()
            # client_trainer.test()

    def read_dataset_ids(self):
        for data in self.head_train_dataset:
            _, _, id_value = data
            self.head_train_ids.append(id_value)

    def print_info(self):
        print(f'MMFL device: {self.device}')
        print(f'Head Model: ClassifierModel')
        print(f'Head Round Num: {self.head_round_num}')
        print(f'Head Dataset: CIFAR')
        print(f'Head Dataset Batch Size: {self.head_dataset_batch_size}')
        print(f'Head Learning rate: {self.head_learn_rate}')
        print(f'Client Num: {self.client_num}')
        print(f'Client Dataset Batch Size: {self.client_dataset_batch_size}')
        print(f'Mini Dataset Size: {self.mini_dataset_size}')
        print(f'Mini Dataset Batch Size: {self.mini_dataset_batch_size}')

    def train(self, epoch):
        print(f'Global train epoch: {epoch}')
        # 随机获取256个样本id
        print(f'    Mini dataset ids generated...')
        random_indices = torch.randperm(len(self.head_train_ids))[:self.mini_dataset_size]
        self.mini_dataset_ids = [self.head_train_ids[i] for i in random_indices]

        # 生成客户端训练嵌入集和测试嵌入集{'client id': {'id': 'embedding'}}
        print(f'    Client embeddings generated...')
        client_train_embeddings = {}
        client_test_embeddings = {}
        for client_id, client_trainer in self.client_trainers.items():
            this_train_embedding, this_test_embedding = client_trainer.generate_client_embedding(self.mini_dataset_batch_size, self.mini_dataset_ids)
            client_train_embeddings[client_id] = this_train_embedding
            client_test_embeddings[client_id] = this_test_embedding

        # print(client_train_embeddings)

        print(f'    Head is training...')
        mini_dataloader = generate_mini_dataloader(self.head_train_dataset, self.mini_dataset_batch_size, self.mini_dataset_ids)

        self.head.train()
        epoch_train_loss_list = []
        for server_epoch in range(self.head_round_num):
            for i, data in enumerate(mini_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                inputs = None
                for _, client_train_embedding in client_train_embeddings.items():
                    if inputs is None:
                        inputs = torch.stack([client_train_embedding[id_value.item()] for id_value in ids], dim=0)
                    else:
                        inputs = torch.cat((inputs, torch.stack([client_train_embedding[id_value.item()] for id_value in ids], dim=0)), dim=1)
                inputs = inputs.to(self.device)
                print(inputs)
                outputs = self.head(inputs)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                l2_loss = self.head.l2_regularization_loss()
                loss += self.weight_decay*l2_loss
                loss.backward()
                epoch_train_loss_list.append(loss.item())
                self.optimizer.step()
            # self.scheduler.step()
        # print(f'    Head train loss: {epoch_train_loss_list}')
        print(f'    Head train loss avg: {sum(epoch_train_loss_list)/len(epoch_train_loss_list)}')
        self.head_train_loss_list.append(sum(epoch_train_loss_list)/len(epoch_train_loss_list))

        print(f'    Head is testing...')
        self.head.eval()
        test_correct = 0
        test_total = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for i, data in enumerate(self.head_test_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                # 根据样本id获取个客户端对应embedding 根据此embedding获得输出并将平均，获取输出结果并判断准确率
                inputs = None
                for _, client_test_embedding in client_test_embeddings.items():
                    if inputs is None:
                        inputs = torch.stack([client_test_embedding[id_value.item()] for id_value in ids], dim=0)
                    else:
                        inputs = torch.cat((inputs, torch.stack([client_test_embedding[id_value.item()] for id_value in ids], dim=0)), dim=1)
                inputs = inputs.to(self.device)
                outputs = self.head(inputs)

                _, predicted = torch.max(outputs.data, 1)
                # print(f'labels: {labels}')
                # print(f'predicted: {predicted}')
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                l2_loss = self.head.l2_regularization_loss()
                total_loss = loss + 0.001 * l2_loss
                epoch_test_loss_list.append(total_loss.item())

        # print(f'    Head test loss: {epoch_test_loss_list}')
        print(f'    Head test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        print(f'    Head history accuracy on test set: {self.head_test_acc_rate_list}')
        print(f'    Head accuracy on test set: {(100 * test_correct / test_total):.2f}%')
        self.head_test_acc_rate_list.append(100 * test_correct / test_total)
        self.head_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

        print(f'    Client is training...')
        for client_id, client_trainer in self.client_trainers.items():
            client_trainer.train(self.head, client_train_embeddings, client_test_embeddings, self.mini_dataset_batch_size, self.mini_dataset_ids)

        del mini_dataloader
        gc.collect()
