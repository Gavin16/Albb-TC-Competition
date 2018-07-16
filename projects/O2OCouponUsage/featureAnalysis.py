# -*-coding:utf-8-*-

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

# import xgboost as xgb
# import lightgbm as lgb


# dfoff = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv')
dfoff = pd.read_csv('../../data/O2OCUF/test.csv')
# dftest = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')
# dfon = pd.read_csv('../../data/O2OCUF/ccf_online_stage1_train.csv')


# dataFrame 默认展示5列，若要展示所有列需要设置 display.max_columns 为 None
pd.set_option('display.max_columns', None)

head10 = dfoff.head(10)
print(type(dfoff.loc[:,'Date_received']))

print(type(dfoff))
print(head10['Date_received'])


# print(head10['Coupon_id'] == 'NaN')

# 是否有优惠券以及是否购买商品
print('有优惠券且购买了商品记录条数：', dfoff[(dfoff['Date_received'] != 'NaN') & (dfoff['Date'] != 'NaN')].shape[0])
print('无优惠券且购买了商品记录条数：', dfoff[(dfoff['Date_received'] == 'NaN') & (dfoff['Date'] != 'NaN')].shape[0])
print('有优惠券且未购买商品记录条数：', dfoff[(dfoff['Date_received'] != 'NaN') & (dfoff['Date'] == 'NaN')].shape[0])
print('无优惠券且未购买商品记录条数：', dfoff[(dfoff['Date_received'] == 'NaN') & (dfoff['Date'] == 'NaN')].shape[0])



