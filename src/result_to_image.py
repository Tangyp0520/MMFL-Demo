import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import matplotlib.pyplot as plt
from openpyxl import load_workbook

x = list(range(100))


class Line(object):
    data = []
    color = 'r'
    line_style = '-'
    label = 'label'

    def __init__(self, data, color, line_style, label):
        self.data = data
        self.color = color
        self.line_style = line_style
        self.label = label


def draw_line(lines, x_label, y_label, title, file_name):
    for line in lines:
        plt.plot(x, line.data, color=line.color, linestyle=line.line_style, label=line.label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig('../results/images/' + file_name + '.png')
    plt.close()


def read_excel(file_name, col_idx, color, line_style, label):
    workbook = load_workbook('../results/' + file_name + '.xlsx')
    sheet = workbook.active
    data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        data.append(row[col_idx])
    return Line(data, color, line_style, label)


local_gray_train_loss = read_excel('local_gray_train_loss_2024_11_21', col_idx=0, color='g', line_style='-', label='local gray train loss')
local_color_train_loss = read_excel('local_color_train_loss_2024_11_21', col_idx=0, color='g', line_style='-', label='local color train loss')
local_multi_train_loss = read_excel('local_multi_train_loss_2024_11_21', col_idx=0, color='g', line_style='-', label='local multi train loss')

local_gray_test_loss = read_excel('local_gray_test_loss_2024_11_21', col_idx=0, color='g', line_style='-', label='local gray test loss')
local_color_test_loss = read_excel('local_color_test_loss_2024_11_21', col_idx=0, color='b', line_style='-', label='local color test loss')
local_multi_test_loss = read_excel('local_multi_test_loss_2024_11_21', col_idx=0, color='y', line_style='-', label='local multi test loss')
mmfl_test_server_loss = read_excel('MMFL_Test_Loss_2024-11-21', col_idx=0, color='r', line_style='-', label='MMFL test server loss')
mmfl_test_client1_loss = read_excel('MMFL_Test_Loss_2024-11-21', col_idx=1, color='r', line_style='--', label='MMFL test client1 loss')
mmfl_test_client2_loss = read_excel('MMFL_Test_Loss_2024-11-21', col_idx=2, color='r', line_style='--', label='MMFL test client2 loss')
mmfl_proto_test_server_loss = read_excel('MMFL_Proto_Test_Loss_2024-11-21', col_idx=0, color='m', line_style='-', label='MMFL proto test server loss')
mmfl_proto_test_client1_loss = read_excel('MMFL_Proto_Test_Loss_2024-11-21', col_idx=1, color='m', line_style='--', label='MMFL proto test client1 loss')
mmfl_proto_test_client2_loss = read_excel('MMFL_Proto_Test_Loss_2024-11-21', col_idx=2, color='m', line_style='--', label='MMFL proto test client2 loss')

local_gray_test_acc = read_excel('local_gray_test_acc_2024_11_21', col_idx=0, color='g', line_style='-', label='local gray test acc')
local_color_test_acc = read_excel('local_color_test_acc_2024_11_21', col_idx=0, color='b', line_style='-', label='local color test acc')
local_multi_test_acc = read_excel('local_multi_test_acc_2024_11_21', col_idx=0, color='y', line_style='-', label='local multi test acc')
mmfl_test_server_acc = read_excel('MMFL_Test_ACC_2024-11-21', col_idx=0, color='r', line_style='-', label='MMFL test server acc')
mmfl_test_client1_acc = read_excel('MMFL_Test_ACC_2024-11-21', col_idx=1, color='r', line_style='--', label='MMFL test client1 acc')
mmfl_test_client2_acc = read_excel('MMFL_Test_ACC_2024-11-21', col_idx=2, color='r', line_style='--', label='MMFL test client2 acc')
mmfl_proto_test_server_acc = read_excel('MMFL_Proto_Test_ACC_2024-11-21', col_idx=0, color='m', line_style='-', label='MMFL proto test server acc')
mmfl_proto_test_client1_acc = read_excel('MMFL_Proto_Test_ACC_2024-11-21', col_idx=1, color='m', line_style='--', label='MMFL proto test client1 acc')
mmfl_proto_test_client2_acc = read_excel('MMFL_Proto_Test_ACC_2024-11-21', col_idx=2, color='m', line_style='--', label='MMFL proto test client2 acc')

loss_lines = [local_gray_test_loss, local_color_test_loss, local_multi_test_loss,
              mmfl_test_server_loss, mmfl_test_client1_loss, mmfl_test_client2_loss,
              mmfl_proto_test_server_loss, mmfl_proto_test_client1_loss, mmfl_proto_test_client2_loss]
acc_lines = [local_gray_test_acc, local_color_test_acc, local_multi_test_acc,
             mmfl_test_server_acc, mmfl_test_client1_acc, mmfl_test_client2_acc,
             mmfl_proto_test_server_acc, mmfl_proto_test_client1_acc, mmfl_proto_test_client2_acc]
draw_line(loss_lines, x_label='Round', y_label='Loss', title='Loss Curve', file_name='cifar_model_fusion_loss_curve100')
draw_line(acc_lines, x_label='Round', y_label='Acc', title='Accuracy Curve', file_name='cifar_model_fusion_acc_curve100')
