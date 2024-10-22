import argparse

from src.datasets.DataloaderGenerator import *
from src.algorithms.MMFL import *

parser = argparse.ArgumentParser(description='MMFL')
args = parser.parse_args()

if __name__ == '__main__':
    print("This is MMFL Demo")
    mmfl = MMFl()
    for epoch in range(100):
        mmfl.train(epoch)

