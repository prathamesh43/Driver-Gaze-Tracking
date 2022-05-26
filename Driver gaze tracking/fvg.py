import numpy as np
import math
import pandas as pd

import torch
from torch.autograd import Variable
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


stk = 'BAJFINANCE'
stocks = pd.read_csv( stk + "_5m_samco.csv")
stocks_data = stocks.iloc[:,1:7].round(2)
print(stocks_data.shape)


def train_test_split(stocks, divide=0.8):
    eightyyy = round(stocks.shape[0] * divide)

    trainn = stocks[:eightyyy]
    testt = stocks[eightyyy:].reset_index(drop=True)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(trainn.shape[0] - 76):
        if trainn['dateTime'][i][11:19] == "09:15:00":
            jrainn = trainn[i:]
            count1200 = 0
            count1000 = 0
            listtxx = []
            highh = []
            loww = []
            for j in range(i, trainn.shape[0]):
                if jrainn['dateTime'][j][11:19] == "10:00:00":
                    count1000 = count1000 + 1
                    closee = jrainn['open'][j]
                if jrainn['dateTime'][j][11:19] == "14:00:00":
                    count1200 = count1200 + 1

                if count1000 != 2:
                    listtxx.append([jrainn['close'][j], jrainn['volume'][j]])
                if count1000 == 2:
                    highh.append(jrainn['high'][j])
                    loww.append(jrainn['low'][j])
                if count1200 == 2:
                    x_train.append(listtxx)
                    y_train.append([round((max(highh) - closee)*100/closee, 5), round((min(loww) - closee)*100/closee, 5)])
                    break

    for i in range(testt.shape[0] - 76):
        if testt['dateTime'][i][11:19] == "09:15:00":
            jrainn = testt[i:]
            count1200 = 0
            count1000 = 0
            listtxx = []
            highh = []
            loww = []
            closee = 0
            for j in range(i, testt.shape[0]):
                if jrainn['dateTime'][j][11:19] == "10:00:00":
                    count1000 = count1000 + 1
                    closee = jrainn['open'][j]
                if jrainn['dateTime'][j][11:19] == "14:00:00":
                    count1200 = count1200 + 1
                if count1000 != 2:
                    listtxx.append([jrainn['close'][j], jrainn['volume'][j]])
                if count1000 == 2:
                    highh.append(jrainn['high'][j])
                    loww.append(jrainn['low'][j])
                if count1200 == 2:
                    x_test.append(listtxx)
                    y_test.append([round((max(highh) - closee)*100/closee, 5), round((min(loww) - closee)*100/closee, 5)])
                    break

    return x_train, y_train, x_test, y_test


#X_train, y_train, X_test, y_test = train_test_split(stocks_data)

y_train = [[[1],[2]],[[1],[2]],[[1],[2]],[[1],[2]]]

#print(X_train)
print(y_train)

xt = Variable(torch.Tensor(y_train))
# scaler = MinMaxScaler(feature_range=(0,1))
# df1 = scaler.fit_transform(np.array(df).reshape(-1,1))