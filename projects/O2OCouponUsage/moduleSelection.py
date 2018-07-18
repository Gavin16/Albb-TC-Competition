# -*-coding:utf-8-*-

# 鉴于特征分析需要花大量的时间处理特征构造数据
# 在模型比较和选择上, 这里从特征分析完成后保存的结果中获取数据

import os, sys, pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb

dfoff = pd.read_csv('../../data/runtimedata/dfoff1.csv')
weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]

### 使用 discount,distance,weekday
### 将数据集按时间分为 训练集(train) 和验证集(valid)
### 20160101 到 20160515 作为train, 20160516 到 20160615作为验证集
### stochastic gradient descent(SGD) 随机梯度下降算法的线性分类模型

# 数据划分， 去掉没有领取优惠券的样本
df = dfoff[dfoff['label'] != -1].copy()
train = df[df['Date_received'] < '20160516'].copy()
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
# 训练集和验证集各包含了多少记录
print(train['label'].value_counts())
print(valid['label'].value_counts())

# 确定需要使用的特征
original_feature = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday',
                    'weekday_type'] + weekdaycols

print(len(original_feature),original_feature)

# model1
predictors = original_feature
print(predictors)

def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])

    return grid_search


if not os.path.isfile('1_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)

