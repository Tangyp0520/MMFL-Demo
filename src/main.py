import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
import datetime

from src.datasets.DataloaderGenerator import *
from src.algorithms.SiloTrainer import *
from src.algorithms.HFM import *
from src.utils.ExcelUtil import *

parser = argparse.ArgumentParser(description='MMFL')
args = parser.parse_args()

if __name__ == '__main__':
    print("This is HFM Demo")
    dataset_root_path = '/home/data2/duwenfeng/datasets/MNIST'
    global_round_num = 50

    print('HFM train start...')
    print(f'Global Round Num: {global_round_num}')
    hfm = HFM(dataset_root_path)
    for epoch in range(global_round_num):
        hfm.train(epoch)
    print('HFM train end...')

    print('Save result...')
    file_head_name = '_hfm_resnet_cifar100_prox_'
    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y_%m_%d')

    head_test_loss_lists = hfm.test_loss_list
    excel_file_name = (date_str + file_head_name + 'loss')
    save_acc_to_excel(excel_file_name, head_test_loss_lists, {})

    head_test_acc_rates = hfm.test_acc_rate_list
    excel_file_name = (date_str + file_head_name + 'acc')
    save_acc_to_excel(excel_file_name, head_test_acc_rates, {})
