import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import datetime

from src.models.ResNetForMNIST import *
from src.models.CNNForMNIST import *
from src.models.CNNForCifar import *
from src.datasets.DataloaderGenerator import *
from src.utils.ExcelUtil import *


print('Generate dataloader: MNIST-M')
# dataset_root_path = 'D:\.download\MNIST-M\data\mnist_m'
dataset_root_path = '/home/data2/duwenfeng/datasets/MNIST'
train_dataset, test_dataset = generate_dataset('Cifar-gray', dataset_root_path)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# train_ids = []
# for batch in train_dataloader:
#     _, _, batch_ids = batch
#     for id_value in batch_ids:
#         train_ids.append(id_value.item())
# random_indices = torch.randperm(len(train_ids))[:len(train_ids)]
# mini_dataset_ids = [train_ids[i] for i in random_indices]
# mini_dataloader = generate_mini_dataloader(train_dataloader, 128, mini_dataset_ids, True)

print('Model init: ResNet')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNetForMNIST(True, 10)
# model = CNNForMNIST(True, 10)
model = CNNForCifar(False, 10)
model.to(device)
learn_rate = 0.001
weight_decay = 0.001
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

local_epoch_num = 10
global_epoch_num = 100
print(f'Device: {device}')
print(f'Learning rate: {learn_rate}')
print(f'Global epoch num: {global_epoch_num}')
print(f'Local epoch num: {local_epoch_num}')
print(f'Criterion: CrossEntropyLoss')
print('Optimizer: SGD')
print('Scheduler: CosineAnnealingLR')

train_loss_list = []
test_loss_list = []
test_acc_rate_list = []

for global_epoch in range(global_epoch_num):
    print(f'Round: {global_epoch:03d}')

    print('    Model is training...')
    model.train()
    epoch_train_loss_list = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        ls_loss = model.l2_regularization_loss()
        loss += weight_decay * ls_loss
        loss.backward()
        optimizer.step()

        epoch_train_loss_list.append(loss.item())
    train_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))
    print('    Train loss: ', sum(epoch_train_loss_list) / len(epoch_train_loss_list))

    print('    Model is testing...')
    model.eval()
    correct = 0
    total = 0
    epoch_test_loss_list = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            l2_loss = model.l2_regularization_loss()
            total_loss = loss + 0.001 * l2_loss
            epoch_test_loss_list.append(total_loss.item())
    print(f'    Test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
    print(f'    Test history accuracy: {test_acc_rate_list}')
    print(f'    Test accuracy: {100 * correct / total:.2f}%')
    test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))
    test_acc_rate_list.append(100 * correct / total)

print(f'Result is saving...')
current_time = datetime.datetime.now()
date_str = current_time.strftime('%Y_%m_%d')
train_loss_excel_name = 'Local_train_loss_'+date_str
test_loss_excel_name = 'Local_test_loss_'+date_str
test_acc_excel_name = 'Local_test_acc_'+date_str

save_acc_to_excel(train_loss_excel_name, train_loss_list, {})
save_acc_to_excel(test_loss_excel_name, test_loss_list, {})
save_acc_to_excel(test_acc_excel_name, test_acc_rate_list, {})


