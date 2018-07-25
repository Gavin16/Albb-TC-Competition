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


# 使用模型4来预测提交数据
df_sub = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_test_revised.csv')
dfoff = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv')

from addFeature import userFeature,merchantFeature,usermerchantFeature,addDiscountFeature,addWeekdayFeature

df_sub1 = addDiscountFeature(df_sub)
df_sub2 = addWeekdayFeature(df_sub1)

feature1 = addDiscountFeature(dfoff)
feature2 = addWeekdayFeature(feature1)

feature = dfoff[
    (dfoff['Date'] < '20160615') | ((dfoff['Date'] == 'null') & (dfoff['Date_received'] < '20160615'))].copy()

user_feature = userFeature(feature2)
merchant_feature = merchantFeature(feature2)
user_merchant_feature = usermerchantFeature(feature2)

# 拼接出预测输入数据
dsub1 = pd.merge(df_sub2,user_feature,on='User_id',how='left').fillna(0)
dsub2 = pd.merge(dsub1,merchant_feature,on='Merchant_id',how='left').fillna(0)
dsub3 = pd.merge(dsub2,user_merchant_feature,on=['User_id', 'Merchant_id'],how='left').fillna(0)

weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
original_feature = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday',
                    'weekday_type'] + weekdaycols

predictors = original_feature + user_feature.columns.tolist()[1:] + \
             merchant_feature.columns.tolist()[1:] + \
             user_merchant_feature.columns.tolist()[2:]

with open('4_model.pkl', 'rb') as f:
    model = pickle.load(f)

dsub_final = dsub3.copy()
dsub_final['Probability'] = model.predict_proba(dsub3[predictors])[:,1]

dsub_submit = dsub_final[['User_id','Coupon_id','Date_received','Probability']]
dsub_submit.to_csv('result_sub1.csv',index=False)
