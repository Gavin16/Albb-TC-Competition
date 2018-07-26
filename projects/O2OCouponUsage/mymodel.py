# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split

# 训练数据
off_train_df = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv')
# 测试数据：提交用
sub_test_df = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')

pd.set_option('display.width', 1000)
print(off_train_df.head(30))
print(off_train_df.describe(include='all'))

# 查看用户数量, 商户数量, 用户消费的商户数量, 来商户消费次数
train_df = off_train_df.copy()
print('用户总记录数为：', train_df.shape[0])
print('不同用户数量为：', len(train_df['User_id'].unique()))

print('商户总记录数为：', train_df.shape[0])
print('不同商户数为：', len(train_df['Merchant_id'].unique()))

userbuydate = train_df[train_df['Date'] != 'null'][['User_id', 'Date']].groupby(['User_id'], as_index=False).count()
userbuydate.columns = ['User_id', 'count']
print('不同用户消费的次数统计：')
print(userbuydate.head(30))

user_receive_coupon = train_df[train_df['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'],as_index=False).count()
user_receive_coupon.columns = ['Date_received', 'count']
print('领券用户领券数量统计：')
print(user_receive_coupon.head(20))

user_buy_with_coupon = train_df[(train_df['Date'] != 'null') & (train_df['Date_received'] != 'null')][
    ['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
user_buy_with_coupon.columns = ['Date_received', 'count']
print('用券用户用券消费的次数统计：')
print(user_buy_with_coupon.head(20))

merchantbuycount = train_df[(train_df['Date'] != 'null')][['Merchant_id', 'Date']].groupby(['Merchant_id'],
                                                                                           as_index=False).count()
merchantbuycount.columns = ['Merchant_id', 'count']
print('不同商户消费次数统计:')
print(merchantbuycount)

# 用户用券消费占领券数的比例
have_counpon_and_use_rate = user_buy_with_coupon['count'] / user_receive_coupon['count']

####  画图查看用户领券后的使用情况
sns.set_style('ticks')
sns.set_context('notebook', font_scale=1.2)
plt.figure(figsize=(10, 6))
# 时间轴
date_received = train_df['Date_received'].unique()
date_received = sorted(date_received[date_received != 'null'])

date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

plt.bar(date_received_dt, user_receive_coupon['count'], label='number of coupon received')
plt.bar(date_received_dt, user_buy_with_coupon['count'], label='number of coupon used')
# 纵轴使用log的度量
plt.yscale('log')
plt.ylabel('Count')
plt.xlabel('date')
plt.legend()
plt.show()


####  提取单个特征,单个特征中挖掘

# 分析优惠折扣, 提取折扣信息
def getDiscountType(row):
    if ':' in row:
        return 1
    if '.' in row:
        return 0
    return 'null'

def getDiscountExpire(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountAmount(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def getDiscountRate(row):
    if '.' in row:
        return float(row)
    if ':' in row:
        rows = row.split(':')
        rate = 1 - float(rows[1])/float(rows[0])
        return rate
    else:
        return 1.0


def addDiscountFeatures(df):
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_expire'] = df['Discount_rate'].apply(getDiscountExpire)
    df['discount_amount'] = df['Discount_rate'].apply(getDiscountAmount)
    df['discount_rate'] = df['Discount_rate'].apply(getDiscountRate)
    return df


# 增加工作日特征
# 获取某一天对应是星期几,1-周一 ... 7-周日
def getWeekdayInfo(row):
    if row != 'null':
        return date(int(row[0:4]),int(row[4:6]),int(row[6:8])).weekday() + 1
    else:
        return row

# 增加weekday特征,若是星期
def addWeekdayFeature(df):
    df['weekday'] = df['Date_received'].astype(str).apply(getWeekdayInfo)
    df['weekday_type'] = df['weekday'].apply(lambda x:1 if x in [6,7] else 0)

    weekday_cols = ['weekday_'+ str(i) for i in range(1,8)]

    # weekday 转化为
    wkdummy = pd.get_dummies(df['weekday'].replace('null',np.nan))
    wkdummy.columns = weekday_cols
    df[weekday_cols] = wkdummy
    return df

# 为数据集添加label, label = 1 代表领取优惠券且已消费,  label = 0代表领取优惠券但是为消费
# label = -1 代表未领取优惠券； label需反映出优惠券对消费的影响
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        #若领取优惠券且也已消费,则判断消费时间是否在领券后的15天以内
        td = pd.to_datetime(row['Date'],format='%Y%m%d') - pd.to_datetime(row['Date_received'])
        if td < pd.Timedelta(15,'D'):
            return 1
    else:
        return 0

def addLabel(df):
    df['label'] = df.apply(label,axis=1)
    return df

train_df = addLabel(train_df)
train_df = addDiscountFeatures(train_df)
train_df = addWeekdayFeature(train_df)
print(train_df.head(20))

####  提取用户特征,商户特征,用户商户组合特征以及用户优惠券组合特征
####  从用户或者商户历史数据中统计
## 将训练集划分为feature 和 data两部分， feature部分用来提取用户特征,商户特征以及用户商户联合特征
## data用来将数据划分为训练集和验证集, 通过计算验证集中模型的AUC值评估模型的性能

# 划分数据集时  将领券时间在20160516之前没有消费的  以及消费时间在20160516之前的数据用来提取特征
feature = train_df[(train_df['Date'] < '20160516')|((train_df['Date'] == 'null') & (train_df['Date_received'] < '20160516'))]
# 领券时间在20160516 到 20160615 之间的数据作为训练/验证集
data = train_df[(train_df['Date_received'] > '20160516') & (train_df['Date_received'] < '20160615')]

# 使用事先写好的添加用户-商户特征的方法
from addFeature import userFeature,merchantFeature,usermerchantFeature
userfeature = userFeature(feature)
merchantfeature = merchantFeature(feature)
usermerchantFeature = usermerchantFeature(feature)

data_tv = pd.merge(data,userfeature,how='left',on='User_id')
data_tv = pd.merge(data_tv,merchantfeature,how='left',on='Merchant_id')
data_tv = pd.merge(data_tv,usermerchantFeature,how='left',on=['User_id','Merchant_id'])

# data 划分成训练集 + 验证集
data = train_test_split(data_tv,stratify=data_tv['label'],random_state=100)














##### 模型选择
# 使用随机梯度下降的 SGDClassifier
# 使用 GBDT(Gradient Boosting Decision Tree)、RandomForest、LR(linear Regression)训练新的模型











