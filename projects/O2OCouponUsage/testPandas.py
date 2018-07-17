# -*-coding:utf-8-*-

import pandas as pd

df = pd.read_csv('../../data/O2OCUF/ccf_offline_stage1_train.csv')

# dtypes
# print(df.dtypes)
# print(df.describe())
# transpose
# print(df.transpose)

df_section = df[0:30]
# print(df_section)
# df_section.to_csv()
df_section[['User_id','Date_received']]










