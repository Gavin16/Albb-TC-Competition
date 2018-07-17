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
dfoff = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv',na_values='NULL')
dftest = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')
# dfon = pd.read_csv('../../data/O2OCUF/ccf_online_stage1_train.csv')


# dataFrame 默认展示5列，若要展示所有列需要设置 display.max_columns 为 None
pd.set_option('display.max_columns', None)
head10 = dfoff.head(10)
print(type(dfoff.loc[:, 'Date_received']))

print(type(dfoff))
print(head10['Date_received'])
print(pd.isnull(head10['Date_received']))


print(dfoff[pd.isnull(dfoff['Date_received']) & pd.isnull(dfoff['Date'])])
print('shape of dfoff is: %d * %d' % (dfoff.shape[0], dfoff.shape[1]))


# 数据集中的null若读取出来为字符串'null' 则可以直接用'null'来比较. 若读取出来为NaN 则需要使用pandas.DataFrame.isnull()来判断
print('有优惠券，购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
print('无优惠券，购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
print('有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

# 测试集中新增用户和商家
print('测试集中出现但训练集中未出现的用户：',set(dftest['User_id']) - set(dfoff['User_id']))
print('测试集中出现但训练集中未出现的商家：',set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))

# Discount_rate拆分成折扣率以及满多少减多少金额




