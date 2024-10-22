import argparse

from src.datasets.DataloaderGenerator import *
from src.algorithms.MMFL import *

parser = argparse.ArgumentParser(description='MMFL')
args = parser.parse_args()

if __name__ == '__main__':
    print("This is MMFL Demo")
    # mnist_train_dataloader, mnist_test_dataloader = generate_dataloader('MNIST', 64)
    # mnistm_train_dataloader, mnistm_test_dataloader = generate_dataloader('MNIST-M', 64)
    mmfl = MMFl()
    for epoch in range(100):
        mmfl.train()

