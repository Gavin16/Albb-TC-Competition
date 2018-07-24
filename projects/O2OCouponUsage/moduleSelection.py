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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dfoff = pd.read_csv('../../data/runtimedata/dfoff1.csv')
dftest = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')
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

print(len(original_feature), original_feature)

# model1
predictors = original_feature
print(predictors)

print(train.head())


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
        n_jobs=1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])

    return grid_search


# 将模型序列化到文件中
if not os.path.isfile('1_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    # 训练集和验证集的划分是按固定的时间划分的,因此第一次训练好的模型之后可以直接用
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)

#
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

y_valid_pred = model.predict_proba(valid[predictors])
print(y_valid_pred)
valid1 = valid.copy()
#  y_valid_pred 作为预测结果, 第一列和第二列, 哪一列是正例的概率？
valid1['pred_prob'] = y_valid_pred[:, 1]
print(valid1.head(2))

# 计算模型验证集的AUC
vg = valid1.groupby(['Coupon_id'])
aucs = []

print(vg['Coupon_id'].value_counts())

for i in vg:
    tmpdf = i[1]
    # print(tmpdf)
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

# 通过客户和商户以前的买卖情况,提取各自或者交叉的特征。
# 选择哪个时间段的数据进行特征提取是可以探索的，这里使用
# 20160101 到20160515之间的数据提取特征，20160516 - 20160615的数据作为训练集
feature = dfoff[
    (dfoff['Date'] < '20160516') | ((dfoff['Date'] == 'null') & (dfoff['Date_received'] < '20160516'))].copy()
data = dfoff[(dfoff['Date_received'] >= '20160516') & (dfoff['Date_received'] <= '20160615')].copy()
print(data['label'].value_counts())

#######  加入用户信息 ########
fdf = feature.copy()
u = fdf[['User_id']].copy().drop_duplicates()
# u_coupon_count:每个用户领取的优惠券的数量
u1 = fdf[fdf['Date_received'] != 'null'][['User_id']].copy()
u1['u_coupon_count'] = 1
u1 = u1.groupby(['User_id'], as_index=False).count()
print(u1.head())

# u_buy_count：用户线下购买次数(用券或者未用券)
u2 = fdf[fdf['Date'] != 'null'][['User_id']].copy()
u2['u_buy_count'] = 1
u2 = u2.groupby(['User_id'], as_index=False).count()
print(u2.head())

# u_buy_with_coupon: 用户线下购买次数(用券)
u3 = fdf[((fdf['Date'] != 'null') & (fdf['Date_received'] != 'null'))][['User_id']].copy()
u3['u_buy_with_coupon'] = 1
u3 = u3.groupby(['User_id'], as_index=False).count()
print(u3.head())

# u_merchant_count: 所有使用优惠券消费的商户
u4 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
u4.drop_duplicates(inplace=True)
u4 = u4.groupby(['User_id'], as_index=False).count()
u4.rename(columns={'Merchant_id': 'u_merchant_count'}, inplace=True)
print(u4.head())

# u_min_distance: 用户到店的距离
utmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['User_id', 'distance']].copy()
utmp.replace(-1, np.nan, inplace=True)
# 用户去店消费的距离的最小值
u5 = utmp.groupby(['User_id'], as_index=False).min()
u5.rename(columns={'distance': 'u_min_distance'}, inplace=True)
print(u5.head())
# 用户去店消费的距离的最大值
u6 = utmp.groupby(['User_id'], as_index=False).max()
u6.rename(columns={'distance': 'u_max_distance'}, inplace=True)
print(u6.head())
# 用户去店消费的距离的平均值
u7 = utmp.groupby(['User_id'], as_index=False).mean()
u7.rename(columns={'distance': 'u_mean_distance'}, inplace=True)
print(u7.head())
# 用户去点消费距离的中位值
u8 = utmp.groupby(['User_id'], as_index=False).median()
u8.rename(columns={'distance': 'u_median_distance'}, inplace=True)
print(u8.head())

# u.shape, u1.shape, u2.shape, u3.shape, u4.shape, u5.shape, u6.shape, u7.shape, u8.shape

# merge all the features on key User_id
user_feature = pd.merge(u, u1, on='User_id', how='left')
user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
user_feature = pd.merge(user_feature, u5, on='User_id', how='left')
user_feature = pd.merge(user_feature, u6, on='User_id', how='left')
user_feature = pd.merge(user_feature, u7, on='User_id', how='left')
user_feature = pd.merge(user_feature, u8, on='User_id', how='left')

# 领券用户用券消费占总所有领券中的比例
user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
    'u_coupon_count'].astype('float')
# 领券用户用券的消费占总消费的比率
user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
    'u_buy_count'].astype('float')
user_feature = user_feature.fillna(0)
print(user_feature.head())

# 将从feature数据集中获取的用户特征以User_id作为连接点 融入data中
data2 = pd.merge(data, user_feature, on='User_id', how='left').fillna(0)

# split data2 into valid and train
train, valid = train_test_split(data2, test_size=0.2, stratify=data2['label'], random_state=100)

# 保存model2
predictors = original_feature + user_feature.columns.tolist()[1:]
print(len(predictors), predictors)

if not os.path.isfile('2_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('2_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('2_model.pkl', 'rb') as f:
        model = pickle.load(f)

# valid set performance
# 直接使用valid做修改会报 SettingWithCopyWarning 警告，因此先做一个备份
valid2 = valid.copy()
valid_predict_proba = model.predict_proba(valid2[predictors])[:, 1]
valid2['pred_prob'] = valid_predict_proba
validgroup = valid2.groupby(['Coupon_id'])

aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)

    aucs.append(auc(fpr, tpr))
    # aucs.append(roc_auc_score(tmpdf['label'], tmpdf['pred_prob']))
print(np.average(aucs))

#######  加入商户信息 ########
# 所有使用优惠券消费中，统计每次消费到商户的距离
m = fdf[['Merchant_id']].copy().drop_duplicates()

# m_coupon_count : 每个商户被领的券的数量
m1 = fdf[fdf['Date_received'] != 'null'][['Merchant_id']].copy()
m1['m_coupon_count'] = 1
m1 = m1.groupby(['Merchant_id'], as_index=False).count()
m1.head()

# m_sale_count : 每个商户到店消费的次数
m2 = fdf[fdf['Date'] != 'null'][['Merchant_id']].copy()
m2['m_sale_count'] = 1
m2 = m2.groupby(['Merchant_id'], as_index=False).count()
m2.head()

# m_sale_with_coupon : 每个商户到店用券消费的次数
m3 = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id']].copy()
m3['m_sale_with_coupon'] = 1
m3 = m3.groupby(['Merchant_id'], as_index=False).count()
m3.head()

mtmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
mtmp.replace(-1, np.nan, inplace=True)
# 以商户分类,到商户消费中距离最小的; 到店消费距离的最小值
m4 = mtmp.groupby(['Merchant_id'], as_index=False).min()
m4.rename(columns={'distance': 'm_min_distance'}, inplace=True)
print("m4", m4.head())
# 商户分组, 所有到商户的消费中距离最大的; 反应到店消费的距离的最大值
m5 = mtmp.groupby(['Merchant_id'], as_index=False).max()
m5.rename(columns={'distance': 'm_max_distance'}, inplace=True)
print("m5", m5.head())
# 商户分组, 所有到商户的消费中距离的平均值; 反应到店消费中距离的平均值
m6 = mtmp.groupby(['Merchant_id'], as_index=False).mean()
m6.rename(columns={'distance': 'm_mean_distance'}, inplace=True)
print("m6", m6.head())
# 商户分组, 所有商户中距离的中位值
m7 = mtmp.groupby(['Merchant_id'], as_index=False).median()
m7.rename(columns={'distance': 'm_median_distance'}, inplace=True)
print("m7", m7.head())

# m.shape, m1.shape, m2.shape, m3.shape, m4.shape, m5.shape, m6.shape, m7.shape
merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m5, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m6, on='Merchant_id', how='left')
merchant_feature = pd.merge(merchant_feature, m7, on='Merchant_id', how='left')
merchant_feature = merchant_feature.fillna(0)
merchant_feature.head(5)

merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
    'm_coupon_count'].astype('float')
merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
    'm_sale_count'].astype('float')
merchant_feature = merchant_feature.fillna(0)
merchant_feature.head()

# add merchant feature to data2
# 在data , user_feature 左侧再加入 merchant_feature 特征
data3 = pd.merge(data2, merchant_feature, on='Merchant_id', how='left').fillna(0)

# split data3 into train/valid
train, valid = train_test_split(data3, test_size=0.2, stratify=data3['label'], random_state=100)

predictors = original_feature + user_feature.columns.tolist()[1:] + merchant_feature.columns.tolist()[1:]
print(predictors)

if not os.path.isfile('3_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('3_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('3_model.pkl', 'rb') as f:
        model = pickle.load(f)

valid3 = valid.copy()
valid3['pred_prob'] = model.predict_proba(valid3[predictors])[:, 1]
validgroup = valid3.groupby(['Coupon_id'])

aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))


# 使用User 和 Merchant 联合特征值做分析
um = fdf[['User_id', 'Merchant_id']].copy().drop_duplicates()

um1 = fdf[['User_id', 'Merchant_id']].copy()
um1['um_count'] = 1
um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()
print(um1.head())

um2 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
um2['um_buy_count'] = 1
um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()
print(um2.head())

um3 = fdf[fdf['Date_received'] != 'null'][['User_id', 'Merchant_id']].copy()
um3['um_coupon_count'] = 1
um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()
print(um3.head())

um4 = fdf[(fdf['Date_received'] != 'null') & (fdf['Date'] != 'null')][['User_id', 'Merchant_id']].copy()
um4['um_buy_with_coupon'] = 1
um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()
print(um4.head())

# merge all user merchant
user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
user_merchant_feature = user_merchant_feature.fillna(0)

user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float')/user_merchant_feature['um_count'].astype('float')
user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_coupon_count'].astype('float')
user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_buy_count'].astype('float')
user_merchant_feature = user_merchant_feature.fillna(0)
user_merchant_feature.head()

data4 = pd.merge(data3, user_merchant_feature, on = ['User_id','Merchant_id'], how = 'left').fillna(0)
train, valid = train_test_split(data4, test_size = 0.2, stratify = data4['label'], random_state=100)

predictors = original_feature + user_feature.columns.tolist()[1:] + \
             merchant_feature.columns.tolist()[1:] + \
             user_merchant_feature.columns.tolist()[2:]
print(len(predictors),predictors)

if not os.path.isfile('4_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('4_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('4_model.pkl', 'rb') as f:
        model = pickle.load(f)

valid4 = valid.copy()
valid4['pred_prob'] = model.predict_proba(valid4[predictors])[:,1]
validgroup = valid4.groupby(['Coupon_id'])

aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))
