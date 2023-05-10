import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys


class MyDatasets(Dataset):
    def __init__(self, day_energy_cost: np.array, feature: np.array):
        super(MyDatasets, self).__init__()
        self.day_energy_cost = day_energy_cost
        self.feature = feature

    def __len__(self):
        return len(self.day_energy_cost)

    def __getitem__(self, index):
        day_energy_cost = self.day_energy_cost[index][1]
        feature = self.feature[index][1:]
        time = torch.arange(0, 96 * 15, 15)
        return time, torch.Tensor(day_energy_cost), torch.Tensor(list(feature))


def data_preprocess(data_path):
    energy_cost = pd.DataFrame(pd.read_csv(data_path + '\\附件1-区域15分钟负荷数据.csv', encoding='ISO-8859-1'))
    energy_cost.rename(columns={'Êý¾ÝÊ±¼ä': '日期', '×ÜÓÐ¹¦¹¦ÂÊ£¨kw£©': '总有功功率'}, inplace=True)
    energy_cost.dropna(inplace=True)

    date = []
    for i in range(len(energy_cost['日期'])):
        date.append(energy_cost['日期'][i].split(' ')[0])
    date = pd.Series(date).drop_duplicates()

    # 集合每天的数据
    print('Collecting energy cost in day ... ')
    day_energy_cost = dict()
    flag = 0
    for day in date:
        day_cost = []
        for day_index in range(flag, len(energy_cost['日期'])):
            if day in energy_cost['日期'][day_index]:
                day_cost.append(energy_cost['总有功功率'][day_index])

            else:
                flag = day_index
                break
        day_energy_cost[day] = day_cost
    # 修复损失的数据
    print('Fixing the lose data ...')
    for day_cost in day_energy_cost.values():
        if len(day_cost) < 96:
            for k in range(96 - len(day_cost)):
                fix = sum(day_cost) / len(day_cost)
                day_cost.append(fix)
    day_energy_cost = pd.DataFrame({'日期': day_energy_cost.keys(), '总有功功率': day_energy_cost.values()})

    info = pd.DataFrame(pd.read_csv(data_path + '\\附件3-气象数据.csv'))
    # 去除重复元数据
    info.drop_duplicates(subset=['日期'], inplace=True)
    info = info.reset_index(drop=True)
    # 对信息进行编码
    # 天气状况
    # 获取所有天气类型并设定对应的编码并存储在weather.txt
    if not os.path.exists(r'C:\Users\13632\Documents\Python_Scripts\研发部机器学习\4.23题目\data\weather.txt'):
        print('Reading all weather situtation ...')
        weather_kinds = []
        for weather_index in range(len(info['天气状况'])):
            weather = info['天气状况'][weather_index]
            for w in weather.split('/'):
                if w not in weather_kinds:
                    weather_kinds.append(w)
        with open(data_path + '\\weather.txt', 'w', encoding='utf-8') as f:
            for w in weather_kinds:
                f.write(w)
                f.write('\n')
    # 对天气状况数据进行编码
    print('Encoding weather situtation ...')
    weather_code = dict()
    with open(data_path + '\\weather.txt', 'r', encoding='utf-8') as f:
        for i, w in enumerate(f.readlines()):
            w = w.strip()
            weather_code[w] = i
    for weather_index in range(len(info['天气状况'])):
        weather = info['天气状况'][weather_index]
        w_code = 0
        for w in weather.split('/'):
            w_code += weather_code[w]
        info['天气状况'][weather_index] = w_code

    # 对最高气温编码
    print('Encoding highest temperature ...')
    for high_t_index in range(len(info['最高温度'])):
        t = info['最高温度'][high_t_index]
        t = t[:-1]
        info['最高温度'][high_t_index] = int(t)
    # 对最低气温编码
    print('Encoding lowest temperature ...')
    for high_t_index in range(len(info['最低温度'])):
        t = info['最低温度'][high_t_index]
        t = t[:-1]
        info['最低温度'][high_t_index] = int(t)

    # 时间编码
    year = []
    month = []
    day = []
    for time_index in range(len(info['日期'])):
        time = info['日期'][time_index]
        time = time.replace('年', '/')
        time = time.replace('月', '/')
        time = time.replace('日', '').split('/')
        year.append(int(time[0]))
        month.append(int(time[1]))
        day.append(int(time[2]))
    info.insert(loc=1, column='年', value=year)
    info.insert(loc=2, column='月', value=month)
    info.insert(loc=3, column='日', value=day)

    return np.array(day_energy_cost), np.array(info.iloc[:, :7])


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\研发部机器学习\4.23题目\data'
    day_energy_cost, feature = data_preprocess(data_path)
    test_dataset = MyDatasets(day_energy_cost, feature)
    time, day_energy_cost_, feature_ = test_dataset.__getitem__(1)
    print(len(time), len(day_energy_cost_), len(feature_))
