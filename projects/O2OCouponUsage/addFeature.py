# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from datetime import date

def userFeature(df):
    u = df[['User_id']].copy().drop_duplicates()

    # u_coupon_count : num of coupon received by user
    u1 = df[df['Date_received'] != 'null'][['User_id']].copy()
    u1['u_coupon_count'] = 1
    u1 = u1.groupby(['User_id'], as_index=False).count()

    # u_buy_count : times of user buy offline (with or without coupon)
    u2 = df[df['Date'] != 'null'][['User_id']].copy()
    u2['u_buy_count'] = 1
    u2 = u2.groupby(['User_id'], as_index=False).count()

    # u_buy_with_coupon : times of user buy offline (with coupon)
    u3 = df[((df['Date'] != 'null') & (df['Date_received'] != 'null'))][['User_id']].copy()
    u3['u_buy_with_coupon'] = 1
    u3 = u3.groupby(['User_id'], as_index=False).count()

    # u_merchant_count : num of merchant user bought from
    u4 = df[df['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace=True)
    u4 = u4.groupby(['User_id'], as_index=False).count()
    u4.rename(columns={'Merchant_id': 'u_merchant_count'}, inplace=True)

    # u_min_distance
    utmp = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['User_id', 'distance']].copy()
    utmp.replace(-1, np.nan, inplace=True)
    u5 = utmp.groupby(['User_id'], as_index=False).min()
    u5.rename(columns={'distance': 'u_min_distance'}, inplace=True)
    u6 = utmp.groupby(['User_id'], as_index=False).max()
    u6.rename(columns={'distance': 'u_max_distance'}, inplace=True)
    u7 = utmp.groupby(['User_id'], as_index=False).mean()
    u7.rename(columns={'distance': 'u_mean_distance'}, inplace=True)
    u8 = utmp.groupby(['User_id'], as_index=False).median()
    u8.rename(columns={'distance': 'u_median_distance'}, inplace=True)

    user_feature = pd.merge(u, u1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u5, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u6, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u7, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u8, on='User_id', how='left')

    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_coupon_count'].astype('float')
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_buy_count'].astype('float')
    user_feature = user_feature.fillna(0)

    print(user_feature.columns.tolist())
    return user_feature


def merchantFeature(df):
    m = df[['Merchant_id']].copy().drop_duplicates()

    # m_coupon_count : num of coupon from merchant
    m1 = df[df['Date_received'] != 'null'][['Merchant_id']].copy()
    m1['m_coupon_count'] = 1
    m1 = m1.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_count : num of sale from merchant (with or without coupon)
    m2 = df[df['Date'] != 'null'][['Merchant_id']].copy()
    m2['m_sale_count'] = 1
    m2 = m2.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_with_coupon : num of sale from merchant with coupon usage
    m3 = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id']].copy()
    m3['m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index=False).count()

    # m_min_distance
    mtmp = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
    mtmp.replace(-1, np.nan, inplace=True)
    m4 = mtmp.groupby(['Merchant_id'], as_index=False).min()
    m4.rename(columns={'distance': 'm_min_distance'}, inplace=True)
    m5 = mtmp.groupby(['Merchant_id'], as_index=False).max()
    m5.rename(columns={'distance': 'm_max_distance'}, inplace=True)
    m6 = mtmp.groupby(['Merchant_id'], as_index=False).mean()
    m6.rename(columns={'distance': 'm_mean_distance'}, inplace=True)
    m7 = mtmp.groupby(['Merchant_id'], as_index=False).median()
    m7.rename(columns={'distance': 'm_median_distance'}, inplace=True)

    merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m5, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m6, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m7, on='Merchant_id', how='left')

    merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
        'm_coupon_count'].astype('float')
    merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / \
                                                  merchant_feature['m_sale_count'].astype('float')
    merchant_feature = merchant_feature.fillna(0)

    print(merchant_feature.columns.tolist())
    return merchant_feature


def usermerchantFeature(df):

    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()

    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um2 = df[df['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    um2['um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um3 = df[df['Date_received'] != 'null'][['User_id', 'Merchant_id']].copy()
    um3['um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um4 = df[(df['Date_received'] != 'null') & (df['Date'] != 'null')][['User_id', 'Merchant_id']].copy()
    um4['um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float')/user_merchant_feature['um_count'].astype('float')
    user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_coupon_count'].astype('float')
    user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_buy_count'].astype('float')
    user_merchant_feature = user_merchant_feature.fillna(0)

    print(user_merchant_feature.columns.tolist())
    return user_merchant_feature


def featureProcess(feature, train, test):
    """
    feature engineering from feature data
    then assign user, merchant, and user_merchant feature for train and test
    """

    user_feature = userFeature(feature)
    merchant_feature = merchantFeature(feature)
    user_merchant_feature = usermerchantFeature(feature)

    train = pd.merge(train, user_feature, on='User_id', how='left')
    train = pd.merge(train, merchant_feature, on='Merchant_id', how='left')
    train = pd.merge(train, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    train = train.fillna(0)

    test = pd.merge(test, user_feature, on='User_id', how='left')
    test = pd.merge(test, merchant_feature, on='Merchant_id', how='left')
    test = pd.merge(test, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    test = test.fillna(0)
    return train, test



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


## 日期转化为星期
def getWeekday(feature):
    if feature == 'null':
        return feature
    else:
        return date(int(feature[0:4]), int(feature[4:6]), int(feature[6:8])).weekday() + 1

# 对读取的dataFrame做处理
def addDiscountFeature(df):
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(getDiscountRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)

    print(df['discount_rate'].unique())

    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    print(df['distance'].unique())
    return df

# 增加星期特征
def addWeekdayFeature(df):
    # datafram 中日优惠券领取时间转星期
    df['weekday'] = df['Date_received'].astype(str).apply(getWeekday)
    # 使用weekday_type 标记是否是周末 0 代表不是，1 代表是
    df['weekday_type'] = df['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    # 将weekday按weekday_1,weekday_2 … weekday_7 字段展示字段为1代表weekday字段就取的该值
    tmpdf = pd.get_dummies(df['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    df[weekdaycols] = tmpdf
    return df




