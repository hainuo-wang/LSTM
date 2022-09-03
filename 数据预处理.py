import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class dataClean():
    filepath = 'C题数据.csv'
    data = pd.read_csv(filepath)
    #  按照日期给数据排序
    data = data.sort_values('接收距离(cm)')
    data.head()
    #  取出价格数据
    price = data[['透气性 mm/s']]
    price.info()

    #  设置归一化区间
    scaler = MinMaxScaler(feature_range=(-1, 1))

    #  将价格归一化（-1，1）
    price['透气性 mm/s'] = scaler.fit_transform(price['透气性 mm/s'].values.reshape(-1, 1))
    #  设置时间序列
    lookback = 2
    #  将价格格式转换成numpy
    data_raw = price.to_numpy()
    data = []
    #  将数据按照序列分组预测
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    #  将数据转换成数组形式
    data = np.array(data)
    #  按行数取20%的数据作为测试集
    test_set_size = int(np.round(0.2 * data.shape[0]))
    #  剩余数据作为训练集
    train_set_size = data.shape[0] - test_set_size
    #  0-train_set_size的行数的数据，且每行中0-（lookback-1）的数据作为x的训练数据集
    #  0-train_set_size的行数的数据，且每行中最后一个数据作为y的训练数据集
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    #  train_set_size-len(data_raw)的行数的数据，且每行中0-（lookback-1）的数据作为x的测试数据集
    #  train_set_size-len(data_raw)的行数的数据，且每行中最后一个数据作为y的训测试数据集
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    # print('x_train.shape = ',x_train.shape)
    # print('y_train.shape = ',y_train.shape)
    # print('x_test.shape = ',x_test.shape)
    # print('y_test.shape = ',y_test.shape)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)