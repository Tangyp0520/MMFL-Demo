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

    print('MMFL train start...')
    print(f'Global Round Num: 100')
    print(f'Dataset Root Path: {dataset_root_path}')
    mmfl = MMFl(dataset_root_path)
    for epoch in range(100):
        mmfl.train(epoch)
    print('MMFL train end.')

    print('Save result...')
    head_train_acc_rates = mmfl.head_train_acc_rates
    client_train_acc_rates = {}
    for client_id, client_trainer in mmfl.client_trainers.items():
        client_train_acc_rates[client_id] = client_trainer.client_train_acc_rates

    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    excel_file_name = ('Train acc ' + date_str
                       + '_HeadRoundNum' + mmfl.head_round_num
                       + '_HeadLearnRate' + mmfl.head_learn_rate)
    save_acc_to_excel(excel_file_name, head_train_acc_rates, client_train_acc_rates)

    head_test_acc_rates = mmfl.head_test_acc_rates
    client_test_acc_rates = {}
    for client_id, client_trainer in mmfl.client_trainers.items():
        client_test_acc_rates[client_id] = client_trainer.client_test_acc_rates

    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    excel_file_name = ('Test acc '+date_str
                       + '_HeadRoundNum' + mmfl.head_round_num
                       + '_HeadLearnRate' + mmfl.head_learn_rate)
    save_acc_to_excel(excel_file_name, head_test_acc_rates, client_test_acc_rates)
