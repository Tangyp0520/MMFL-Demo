import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
import datetime

from src.datasets.DataloaderGenerator import *
from src.algorithms.MMFL import *
from src.utils.ExcelUtil import *

parser = argparse.ArgumentParser(description='MMFL')
args = parser.parse_args()

if __name__ == '__main__':
    print("This is MMFL Demo")
    # dataset_root_path = 'D:\\.download\\ModelNet10\\dataset'
    # dataset_root_path = '/home/data2/duwenfeng/datasets/ModelNet10'
    # dataset_root_path = 'D:\.download\MNIST-M\data\mnist_m'
    dataset_root_path = '/home/data2/duwenfeng/datasets/MNIST'
    global_round_num = 100

    print('MMFL train start...')
    print(f'Global Round Num: {global_round_num}')
    print(f'Dataset Root Path: {dataset_root_path}')
    mmfl = MMFl(dataset_root_path)
    for epoch in range(global_round_num):
        mmfl.train(epoch)
    print('MMFL train end.')

    print('Save result...')
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')

    # head_train_loss_lists = mmfl.train_loss_list
    # client_train_loss_lists = {}
    # for client_id, client_trainer in mmfl.client_trainers.items():
    #     client_train_loss_lists[client_id] = client_trainer.client_train_loss_list
    # excel_file_name = ('Train Loss ' + date_str
    #                    + '_HeadRoundNum' + str(mmfl.head_round_num)
    #                    + '_HeadLearnRate' + str(mmfl.head_learn_rate))
    # save_acc_to_excel(excel_file_name, head_train_loss_lists, client_train_loss_lists)

    head_test_loss_lists = mmfl.test_loss_list
    client_test_loss_lists = {}
    for client_id, client_trainer in mmfl.client_trainers.items():
        client_test_loss_lists[client_id] = client_trainer.client_test_loss_list
    excel_file_name = ('MMFL_Proto_Test_Loss_' + date_str)
                       # + '_HeadRoundNum' + str(mmfl.head_round_num)
                       # + '_HeadLearnRate' + str(mmfl.head_learn_rate))
    save_acc_to_excel(excel_file_name, head_test_loss_lists, client_test_loss_lists)

    head_test_acc_rates = mmfl.test_acc_rate_list
    client_test_acc_rates = {}
    for client_id, client_trainer in mmfl.client_trainers.items():
        client_test_acc_rates[client_id] = client_trainer.client_test_acc_rate_list
    excel_file_name = ('MMFL_Proto_Test_ACC_'+date_str)
                       # + '_HeadRoundNum' + str(mmfl.head_round_num)
                       # + '_HeadLearnRate' + str(mmfl.head_learn_rate))
    save_acc_to_excel(excel_file_name, head_test_acc_rates, client_test_acc_rates)
