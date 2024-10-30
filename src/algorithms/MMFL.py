import gc

import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.algorithms.ClientTrainer import *


class MMFl(object):
    def __init__(self, dataset_root_path):
        self.dataset_root_path = dataset_root_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_round_num = 20
        self.head_dataset_batch_size = 64
        self.head_train_dataloader, self.head_test_dataloader = generate_dataloader('ModelNet10',
                                                                                    self.head_dataset_batch_size,
                                                                                    dataset_root_path + '/view1')
        self.head_train_ids = []
        self.read_dataset_ids()

        self.head = ClassifierModel()
        self.head.to(self.device)
        self.head_learn_rate = 0.01
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.head.parameters(), lr=self.head_learn_rate, momentum=0.9,
                                         weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.client_num = 4
        self.client_dataset_batch_size = 64
        self.client_trainers = {}
        self.client_ids = []
        for i in range(self.client_num):
            train_dataloader, test_dataloader = generate_dataloader('ModelNet10',
                                                                    self.client_dataset_batch_size,
                                                                    dataset_root_path + '/view' + str(i + 1))
            self.client_trainers[i] = ClientTrainer(i, self.head, train_dataloader, test_dataloader)
            self.client_ids.append(i)

        self.mini_dataset_size = 256
        self.mini_dataset_ids = []
        self.mini_dataset_batch_size = 64

        self.head_acc_rates = []
        self.print_info()
        for _, client_trainer in self.client_trainers.items():
            client_trainer.print_info()
            # client_trainer.test()

    def read_dataset_ids(self):
        root_dir = self.dataset_root_path + '/view1'
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name, 'train')
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    self.head_train_ids.append(file_name)

    def print_info(self):
        print(f'MMFL device: {self.device}')
        print(f'Head Model: ClassifierModel')
        print(f'Head Round Num: {self.head_round_num}')
        print(f'Head Dataset: ModelNet10')
        print(f'Head Dataset Batch Size: {self.head_dataset_batch_size}')
        print(f'Head Learning rate: {self.head_learn_rate}')
        print(f'Client Num: {self.client_num}')
        print(f'Client Dataset: ModelNet10 view i')
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
        clint_train_embeddings = {}
        clint_test_embeddings = {}
        for client_id, client_trainer in self.client_trainers.items():
            this_train_embedding, this_test_embedding = client_trainer.generate_client_embedding(self.mini_dataset_batch_size, self.mini_dataset_ids)
            clint_train_embeddings[client_id] = this_train_embedding
            clint_test_embeddings[client_id] = this_test_embedding

        # 训练head
        print(f'    Head is training...')
        self.head.train()
        mini_dataloader = generate_mini_dataloader(self.head_train_dataloader, self.mini_dataset_batch_size,
                                                   self.mini_dataset_ids)
        for server_epoch in range(self.head_round_num):
            for i, data in enumerate(mini_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                # 根据样本id获取各客户端对应embedding 根据此embedding获得损失并平均，将平均损失作为整体损失反向传播
                avg_loss = 0
                for _, client_train_embedding in clint_train_embeddings.items():
                    inputs = torch.stack([client_train_embedding[id_value] for id_value in ids], dim=0)

                    self.optimizer.zero_grad()
                    output = self.head(inputs)
                    this_loss = self.criterion(output, labels)
                    avg_loss += this_loss
                avg_loss /= self.client_num
                avg_loss.backward()
                self.optimizer.step()
        self.scheduler.step()

        # 测试head
        print(f'    Head is testing...')
        self.head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.head_test_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                # 根据样本id获取个客户端对应embedding 根据此embedding获得输出并将平均，获取输出结果并判断准确率
                avg_output = None
                for client_id, client_test_embedding in clint_test_embeddings.items():
                    inputs = torch.stack([client_test_embedding[id_value] for id_value in ids], dim=0)

                    this_output = self.head(inputs)
                    if avg_output is None:
                        avg_output = this_output
                    else:
                        avg_output += this_output
                avg_output /= self.client_num
                _, predicted = torch.max(avg_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Round {epoch+1:03d} head accuracy on test set: {(100 * correct / total):.2f}%')
        self.head_acc_rates.append(100 * correct / total)

        print(f'    Client is training...')
        for client_id, client_trainer in self.client_trainers.items():
            client_trainer.train(self.head, clint_train_embeddings, self.mini_dataset_batch_size, self.mini_dataset_ids)

        del mini_dataloader
        gc.collect()
