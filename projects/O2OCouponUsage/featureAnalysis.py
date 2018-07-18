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

import xgboost as xgb
import lightgbm as lgb

# dfoff = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv')
dfoff = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv', na_values='NULL')
dftest = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')
# dfon = pd.read_csv('../../data/O2OCUF/ccf_online_stage1_train.csv')


# dataFrame 默认展示5列，若要展示所有列需要设置 display.max_columns 为 None
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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
print('测试集中出现但训练集中未出现的用户：', set(dftest['User_id']) - set(dfoff['User_id']))
print('测试集中出现但训练集中未出现的商家：', set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))


#### Discount_rate拆分成折扣率以及满多少减多少金额
# Discount_rate 有两种形式：0.A  和 N:M   对应的将这两种类型用discount_type 标记
def getDiscountType(row):
    if ':' in row:
        return 1
    elif '.' in row:
        return 0
    else:
        return 'null'


# 获取优惠券的折扣[0,1.0]
def getDiscountRate(row):
    if ':' in row:
        rows = row.split(':')
        dis_rate = 1 - float(rows[1]) / float(rows[0])
        return dis_rate
    elif '.' in row:
        return float(row)
    else:
        return 1.0


# 获取优惠券需要购满的金额
def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


# 获取购满该金额后优惠金额
def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


# 对读取的dataFrame做处理
def processData(df):
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(getDiscountRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)

    print(df['discount_rate'].unique())

    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    print(df['distance'].unique())
    return df


dfoff = processData(dfoff)
print(dfoff.head())

####  领取优惠券时间和使用时间
date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[date_received != 'null'])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy != 'null'])

# date_buy1 = sorted(dfoff[dfoff['Date'] != 'null']['Date'])
print('优惠券收到日期从 ', date_received[0], '到', date_received[-1])
print('消费日期从', date_buy[0], '到', date_buy[-1])

# 领取了优惠券,不管是否使用
couponbydate = dfoff[dfoff['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'],
                                                                                          as_index=False).count()
couponbydate.columns = ['Date_received', 'count']
# 领取了优惠券且使用了优惠券以Date_received来统计
buybydate = dfoff[(dfoff['Date'] != 'null') & (dfoff['Date_received'] != 'null')][['Date_received', 'Date']].groupby(
    ['Date_received'], as_index=False).count()

buybydate.columns = ['Date_received', 'count']

print(couponbydate)
print(buybydate)

# 画图比较领取福利券中使用的比率
sns.set_style('ticks')
sns.set_context('notebook', font_scale=1.5)
plt.figure(figsize=(12, 8))
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt, couponbydate['count'], label='number of coupon received')
plt.bar(date_received_dt, buybydate['count'], label='number of coupon used')

plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'])
plt.ylabel('Ratio(coupon used/coupon received)')
plt.tight_layout()
plt.show()


## 日期转化为星期
def getWeekday(feature):
    if feature == 'null':
        return feature
    else:
        return date(int(feature[0:4]), int(feature[4:6]), int(feature[6:8])).weekday() + 1


# datafram 中日优惠券领取时间转星期
dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)
# 使用weekday_type 标记是否是周末 0 代表不是，1 代表是
dfoff['weekdat_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
print(weekdaycols)

# 将weekday按weekday_1,weekday_2 … weekday_7 字段展示字段为1代表weekday字段就取的该值
tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

print(dfoff)


## 增加label标签用来记录是否领取,是否使用优惠券
# 若未领取优惠券则返回-1 ，若领取了优惠券但是未使用则返回0，若领取了且使用了优惠券则返回1
def lable(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        # 若消费日期在领取日期之后的15天内，则默认都使用了该优惠券
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

dfoff['label'] = dfoff.apply(lable, axis=1)

print(dfoff['label'].value_counts())

# 保存处理后的数据
dfoff.to_csv('../../data/runtimedata/dfoff1.csv', index=False)
