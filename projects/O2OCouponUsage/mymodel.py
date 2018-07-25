# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

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



# 增加工作日特征



####  提取用户特征,从用户或者商户历史数据中统计







