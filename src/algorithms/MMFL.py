import torch
import torch.nn as nn
from src.models.ClassfierModel import *
from src.models.ModelNetResnet18 import *
from src.datasets.DataloaderGenerator import *
from ClientTrainer import *


class MMFl(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_epoch = 20
        self.global_train_dataloader, self.global_test_dataloader = generate_dataloader('ModelNet10', 64, 'D:\\.download\\ModelNet10\\dataset\\view1')

        self.head = ClassifierModel()
        self.head.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.head.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.client_num = 4
        self.client_trainer = {}
        self.client_ids = []
        for i in range(self.client_num):
            train_dataloader, test_dataloader = generate_dataloader('ModelNet10', 64, 'D:\\.download\\ModelNet10\\dataset\\view' + str(i + 1))
            self.client_trainer[i] = ClientTrainer(i, self.head, train_dataloader, test_dataloader)
            self.client_ids.append(i)

        self.mini_batch_size = 256
        self.mini_batch_ids = []

    def train(self, epoch):
        # 随机获取256个样本id
        all_ids = []
        for batch in self.global_train_dataloader:
            _, _, ids_in_batch = batch
            all_ids.extend(ids_in_batch)
        random_indices = torch.randperm(len(all_ids))[:self.mini_batch_size]
        self.mini_batch_ids = [all_ids[i] for i in random_indices]

        # 生成客户端训练嵌入集和测试嵌入集
        clint_train_embeddings = {}
        clint_test_embeddings = {}
        for client_id, client_trainer in self.client_trainer.items():
            this_train_embedding, this_test_embedding = client_trainer.generate_client_embedding(self.mini_batch_ids)
            clint_train_embeddings[client_id] = this_train_embedding
            clint_test_embeddings[client_id] = this_test_embedding

        # 训练head
        self.head.train()
        mini_dataloader = generate_mini_dataloader(self.global_train_dataloader, 64, self.mini_batch_ids)
        for server_epoch in range(self.global_epoch):
            for i, data in enumerate(mini_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                total_loss = 0
                for client_id, client_train_embedding in clint_train_embeddings.items():
                    data = [client_train_embedding[id_value] for id_value in ids]
                    data = torch.tensor(data).to(self.device)
                    output = self.head(data)

                    self.optimizer.zero_grad()
                    this_loss = self.criterion(output, labels)
                    total_loss += this_loss.item()
                total_loss /= self.client_num
                torch.tensor(total_loss).backward()
                self.optimizer.step()
        self.scheduler.step()

        # 测试head
        self.head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.global_test_dataloader):
                _, labels, ids = data
                labels = labels.to(self.device)
                avg_output = torch.zeros(10, labels.size(0)).to(self.device)
                for client_id, client_test_embedding in clint_test_embeddings.items():
                    data = [client_test_embedding[id_value] for id_value in ids]
                    data = torch.tensor(data).to(self.device)
                    output = self.head(data)
                    avg_output += output
                avg_output /= self.client_num
                _, predicted = torch.max(avg_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Round {epoch+1:03d} accuracy on test set: {(100 * correct / total):.2f}%')

        for client_id, client_trainer in self.client_trainer.items():
            client_trainer.train(self.head, clint_train_embeddings, self.mini_batch_ids)




