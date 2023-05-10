import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def R_square(pred: torch.Tensor, real: torch.Tensor):
    mean = torch.mean(pred)
    with torch.no_grad():
        R = 1 - torch.sum(pow((real - pred), 2)) / torch.sum(pow((real - mean), 2))
    return float(R)


def Visualization(evaluation, train: bool, save_option: str = None):
    index = list(evaluation.keys())
    if train:
        index.remove('epoch')
        epoch = range(1, evaluation['epoch'] + 1)
        for i, k in enumerate(index):
            plt.plot(epoch, evaluation[k][0], label='train')
            plt.plot(epoch, evaluation[k][1], label='val')
            plt.title('train' + k)
            plt.xlabel('epoch')
            plt.ylabel(k)
            plt.legend()
            plt.grid()
            plt.show()

    else:
        index.remove('num')
        num = range(1, evaluation['num'] + 1)
        for i, k in enumerate(index):
            plt.plot(num, evaluation[k], label='test')
            plt.title('test' + k)
            plt.xlabel('num')
            plt.ylabel(k)
            plt.legend()
            plt.grid()
            plt.show()


if __name__ == '__main__':
    pred = torch.rand(1, 96)
    real = torch.rand(1, 96)
    print(R_square(pred, real))
