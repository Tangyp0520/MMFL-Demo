import argparse
import datetime

from src.datasets.DataloaderGenerator import *
from src.algorithms.MMFL import *
from src.utils.ExcelUtil import *

parser = argparse.ArgumentParser(description='MMFL')
args = parser.parse_args()

if __name__ == '__main__':
    print("This is MMFL Demo")
    dataset_root_path = 'D:\\.download\\ModelNet10\\dataset'

    print('MMFL train start...')
    print(f'Global Round Num: 100')
    print(f'Dataset Root Path: {dataset_root_path}')
    mmfl = MMFl(dataset_root_path)
    for epoch in range(100):
        mmfl.train(epoch)
    print('MMFL train end.')

    print('Save result...')
    head_acc_rates = mmfl.head_acc_rates
    client_acc_rates = {}
    for client_id, client_trainer in mmfl.client_trainers.items():
        client_acc_rates[client_id] = client_trainer.client_acc_rates

    current_time = datetime.datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    excel_file_name = (date_str
                       + '_HeadRoundNum' + mmfl.head_round_num
                       + '_HeadLearnRate' + mmfl.head_learn_rate)
    save_acc_to_excel(excel_file_name, head_acc_rates, client_acc_rates)

