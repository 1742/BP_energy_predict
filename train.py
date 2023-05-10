import sys

import torch
from torch import nn
from model.BP import *

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, data_preprocess
import numpy as np

import os
import json
import random
from tqdm import tqdm
from tools.evaluation_index import R_square, Visualization


data_path = r'C:\Users\13632\Documents\Python_Scripts\研发部机器学习\4.23题目\data'
pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\研发部机器学习\4.23题目\model\BP.pth'
save_path = r'C:\Users\13632\Documents\Python_Scripts\研发部机器学习\4.23题目\model'
effect_path = r'runs/effect.json'

learning_rate = 10
weight_decay = 1e-8
epochs = 100
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The train will run in {} ...'.format(device))
pretrained = False
save_option = True


def train(
        device: str,
        model: nn.Module,
        train_datasets: Dataset,
        val_datasets: Dataset,
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        optim: str,
        criterion_name: str,
        pretrained: bool,
        save_option: bool,
        lr_schedule: dict = None
        ):

    # 记录验证集最好状态，并提前结束训练
    # best_val = 0
    # flag = 0

    # 返回指标
    train_loss = []
    train_R_square = []
    val_loss = []
    val_R_square = []

    if pretrained:
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            print('Successfully load pretrained model from {}!'.format(pretrained_path))
        else:
            print('model parameters files is not exist!')
            sys.exit(0)
    model.to(device)

    if batch_size == 1:
        is_batch = False
    else:
        is_batch = True

    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if criterion_name == 'BCELoss':
        criterion = nn.BCELoss()
    elif criterion_name == 'MSELoss' or criterion_name == 'MSE':
        criterion = nn.MSELoss()

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif lr_schedule['name'] == 'StepLR':
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule['step_size'], gamma=lr_schedule['gamma'])
    elif lr_schedule['name'] == 'ExponentialLR':
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schedule['gamma'])

    for epoch in range(epochs):
        # 训练

        # model.train()的作用是启用 Batch Normalization 和 Dropout
        model.train()

        per_train_loss = 0
        per_train_r = 0
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (time, day_energy_cost, feature) in enumerate(train_dataloader):
                time = time.to(device, dtype=torch.float)
                day_energy_cost = day_energy_cost.to(device, dtype=torch.float)
                feature = feature.to(device, dtype=torch.float)

                output = model(time, feature).float()

                loss = criterion(output, day_energy_cost)
                r = R_square(output, day_energy_cost)

                # 记录每批次平均指标
                per_train_loss += loss.item()
                per_train_r += r

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_schedule.step()

                pbar.set_postfix({criterion_name: loss.item(), 'R_square': r})

                pbar.update(1)

        # 记录每训练次平均指标
        train_loss.append(per_train_loss / len(train_dataloader))
        train_R_square.append(per_train_r / len(train_dataloader))

        # 验证
        # model.eval()的作用是禁用 Batch Normalization 和 Dropout
        model.eval()

        per_val_loss = 0
        per_val_r = 0

        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (time, day_energy_cost, feature) in enumerate(val_dataloader):
                    time = time.to(device, dtype=torch.float)
                    day_energy_cost = day_energy_cost.to(device, dtype=torch.float)
                    feature = feature.to(device, dtype=torch.float)

                    output = model(time, feature).float()

                    loss = criterion(output, day_energy_cost)
                    r = R_square(output, day_energy_cost)

                    # 记录指标
                    per_val_loss += loss.item()
                    per_val_r += r
                    pbar.set_postfix({criterion_name: loss.item(), 'R_square': r})

                    pbar.update(1)

        # 记录每训练次平均指标
        val_loss.append(per_val_loss / len(val_dataloader))
        val_R_square.append(per_val_r / len(val_dataloader))

        # record_val_loss = per_val_loss / len(val_dataloader)
        # record_val_acc = per_val_acc / len(val_dataloader)
        # record_val_mIOU = per_val_mIou / len(val_dataloader)

        # 提前结束条件
        # if (record_val_acc + record_val_mIOU - record_val_loss) > best_val:
        #     best_val = per_val_acc + per_val_mIou - per_val_loss
        # if (record_val_acc + record_val_mIOU - record_val_loss) < best_val and record_val_acc > 0.9 and record_val_mIOU > 0.8:
        #     flag = 1
        #
        # if flag:
        #     break

    if save_option:
        torch.save(model.state_dict(), os.path.join(save_path, 'BP.pth'))

    return {'epoch': epoch+1, 'loss': [train_loss, val_loss], 'R_square': [train_R_square, val_R_square]}


if __name__ == '__main__':
    # 划分数据集
    day_energy_cost, feature = data_preprocess(data_path)
    data_num = len(day_energy_cost)
    train_energy_data = day_energy_cost[:int(data_num * 0.7)]
    train_feature = feature[:int(data_num * 0.7)]
    val_energy_data = day_energy_cost[int(data_num * 0.7):int(data_num * 0.9)]
    val_feature = feature[int(data_num * 0.7):int(data_num * 0.9)]
    test_energy_data = day_energy_cost[int(data_num * 0.9):]
    test_feature_data = feature[int(data_num * 0.9):]
    print('train_data_num:', len(train_energy_data))
    print('val_data_num:', len(val_energy_data))
    print('test_data_num:', len(test_energy_data))

    train_datasets = MyDatasets(train_energy_data, train_feature)
    val_datasets = MyDatasets(val_energy_data, val_feature)

    model = BP(102, 96)
    optimizer = 'Adam'
    # criterion = 'BCELoss'
    criterion = 'MSELoss'
    lr_schedule = {'name': 'ExponentialLR', 'gamma': 0.99}
    print('loss:', criterion)
    print('optimizer:', optimizer)
    print('lr_schedule:', lr_schedule)

    effect = train(
        device=device,
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        optim=optimizer,
        criterion_name=criterion,
        pretrained=pretrained,
        save_option=save_option,
        lr_schedule=lr_schedule
    )

    with open(effect_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(effect))

    Visualization(effect, True)
