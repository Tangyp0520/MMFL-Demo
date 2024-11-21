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
from src.models.MultiModelForCifar import *
from src.datasets.DataloaderGenerator import *
from src.utils.ExcelUtil import *


def color_test(train_dataloader, test_dataloader, local_round_num):
    print(f'Color Client')
    # local_round_num = 50
    learning_rate = 0.001
    weight_decay = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModelForCifar(device)
    model.to(device)

    classifier_params = model.classifier.parameters()
    color_params = model.color_model.parameters()
    gray_params = model.gray_model.parameters()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam([
        {'params': classifier_params, 'weight_decay': weight_decay},
        {'params': color_params, 'weight_decay': weight_decay},
        {'params': gray_params, 'weight_decay': weight_decay, 'lr': 0}
    ], lr=learning_rate, weight_decay=weight_decay)
    client_train_loss_list = []
    client_test_loss_list = []
    client_test_acc_rate_list = []

    print(f'    Color client train')
    for epoch in range(local_round_num):
        print(f'    Color client epoch: {epoch}')
        model.train()
        epoch_train_loss_list = []
        for batch in train_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(device), gray.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(color, gray)
            loss = criterion(output, labels)
            loss.backward()
            epoch_train_loss_list.append(loss.item())
            optimizer.step()
        print(f'    Color client train loss avg: {sum(epoch_train_loss_list) / len(epoch_train_loss_list)}')
        client_train_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))
        print(f'    Color client test')
        model.eval()
        total = 0
        correct = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for batch in test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(device), gray.to(device), labels.to(device)
                output = model(color, gray)
                loss = criterion(output, labels)
                epoch_test_loss_list.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(f'labels: {labels}')
                # print(f'predicted: {predicted}')
        print(f'    Color client accuracy on test set: {(100 * correct / total):.2f}%')
        print(f'    Color client test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        client_test_acc_rate_list.append(100 * correct / total)
        client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

    print(f'    Color client result is saving...')
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')
    train_loss_excel_name = 'local_color_train_loss_' + date_str
    test_loss_excel_name = 'local_color_test_loss_' + date_str
    test_acc_excel_name = 'local_color_test_acc_' + date_str

    save_acc_to_excel(train_loss_excel_name, client_train_loss_list, {})
    save_acc_to_excel(test_loss_excel_name, client_test_loss_list, {})
    save_acc_to_excel(test_acc_excel_name, client_test_acc_rate_list, {})


def gray_test(train_dataloader, test_dataloader, local_round_num):
    print(f'Gray Client')
    # local_round_num = 50
    learning_rate = 0.001
    weight_decay = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModelForCifar(device)
    model.to(device)

    classifier_params = model.classifier.parameters()
    color_params = model.color_model.parameters()
    gray_params = model.gray_model.parameters()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam([
        {'params': classifier_params, 'weight_decay': weight_decay},
        {'params': color_params, 'weight_decay': weight_decay, 'lr': 0},
        {'params': gray_params, 'weight_decay': weight_decay}
    ], lr=learning_rate, weight_decay=weight_decay)
    client_train_loss_list = []
    client_test_loss_list = []
    client_test_acc_rate_list = []

    print(f'    Gray client train')
    for epoch in range(local_round_num):
        print(f'    Gray client epoch: {epoch}')
        model.train()
        epoch_train_loss_list = []
        for batch in train_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(device), gray.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(color, gray)
            loss = criterion(output, labels)
            loss.backward()
            epoch_train_loss_list.append(loss.item())
            optimizer.step()
        print(f'    Gray client train loss avg: {sum(epoch_train_loss_list) / len(epoch_train_loss_list)}')
        client_train_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))
        print(f'    Gray client test')
        model.eval()
        total = 0
        correct = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for batch in test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(device), gray.to(device), labels.to(device)
                output = model(color, gray)
                loss = criterion(output, labels)
                epoch_test_loss_list.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    Gray client accuracy on test set: {(100 * correct / total):.2f}%')
        print(f'    Gray client test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        client_test_acc_rate_list.append(100 * correct / total)
        client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

    print(f'    Gray client result is saving...')
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')
    train_loss_excel_name = 'local_gray_train_loss_' + date_str
    test_loss_excel_name = 'local_gray_test_loss_' + date_str
    test_acc_excel_name = 'local_gray_test_acc_' + date_str

    save_acc_to_excel(train_loss_excel_name, client_train_loss_list, {})
    save_acc_to_excel(test_loss_excel_name, client_test_loss_list, {})
    save_acc_to_excel(test_acc_excel_name, client_test_acc_rate_list, {})


def multi_test(train_dataloader, test_dataloader, local_round_num):
    print(f'Multiple Client')
    # local_round_num = 50
    learning_rate = 0.001
    weight_decay = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModelForCifar(device)
    model.to(device)

    classifier_params = model.classifier.parameters()
    color_params = model.color_model.parameters()
    gray_params = model.gray_model.parameters()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam([
        {'params': classifier_params, 'weight_decay': weight_decay},
        {'params': color_params, 'weight_decay': weight_decay},
        {'params': gray_params, 'weight_decay': weight_decay}
    ], lr=learning_rate, weight_decay=weight_decay)
    client_train_loss_list = []
    client_test_loss_list = []
    client_test_acc_rate_list = []

    print(f'    Multiple client train')
    for epoch in range(local_round_num):
        print(f'    Multiple client epoch: {epoch}')
        model.train()
        epoch_train_loss_list = []
        for batch in train_dataloader:
            color, gray, labels, _ = batch
            color, gray, labels = color.to(device), gray.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(color, gray)
            loss = criterion(output, labels)
            loss.backward()
            epoch_train_loss_list.append(loss.item())
            optimizer.step()
        print(f'    Multiple client train loss avg: {sum(epoch_train_loss_list) / len(epoch_train_loss_list)}')
        client_train_loss_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))
        print(f'    Multiple client test')
        model.eval()
        total = 0
        correct = 0
        epoch_test_loss_list = []
        with torch.no_grad():
            for batch in test_dataloader:
                color, gray, labels, _ = batch
                color, gray, labels = color.to(device), gray.to(device), labels.to(device)
                output = model(color, gray)
                loss = criterion(output, labels)
                epoch_test_loss_list.append(loss.item())

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'    Multiple client accuracy on test set: {(100 * correct / total):.2f}%')
        print(f'    Multiple client test loss avg: {sum(epoch_test_loss_list) / len(epoch_test_loss_list)}')
        client_test_acc_rate_list.append(100 * correct / total)
        client_test_loss_list.append(sum(epoch_test_loss_list) / len(epoch_test_loss_list))

    print(f'    Multiple client result is saving...')
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')
    train_loss_excel_name = 'local_multi_train_loss_' + date_str
    test_loss_excel_name = 'local_multi_test_loss_' + date_str
    test_acc_excel_name = 'local_multi_test_acc_' + date_str

    save_acc_to_excel(train_loss_excel_name, client_train_loss_list, {})
    save_acc_to_excel(test_loss_excel_name, client_test_loss_list, {})
    save_acc_to_excel(test_acc_excel_name, client_test_acc_rate_list, {})


train_dataset, test_dataset = generate_dataset('Multiple')
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

local_round_num = 100

multi_test(train_dataloader, test_dataloader, local_round_num)
color_test(train_dataloader, test_dataloader, local_round_num)
gray_test(train_dataloader, test_dataloader, local_round_num)
