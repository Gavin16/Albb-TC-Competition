# -*-coding:utf-8-*-
# 10 minutes to pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 2, 3, 4, np.nan, 5, 7, 9])
print(s)

dates = pd.date_range('20180101', periods=7)

print(dates)

# creat a DataFrame by passing a NumPy array
df = pd.DataFrame(np.random.randn(7, 4), index=dates, columns=list('ABCD'))
print(df)

# creat a DataFrame by passing a dict of Objects
df2 = pd.DataFrame({'A': 1,
                    'B': pd.Timestamp('20180725'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})
print(df2)

print(df2.dtypes)
print(df2.info())
print(df2.tail(2))

print(df2.shape)
print(df2.size)

# 虽然不提示，DataFrame 中包含columns 和 index属性
print(df2.columns)
print(df2.index)
print(df2.values)

print(df2.describe())

# transpose
print(df2.T)

# sort by an axis, DataFrame will be sorted by column name descendingly
df3 = df.sort_index(axis=1, ascending=False)
print(df3)

# sort by value, a column of DataFrame will be sorted descending
df4 = df.sort_values(by='A', ascending=False)
print(df4)

# getting, tow ways to get a column
print(df['A'])
print(df.A)

# use [] to slice the rows
print(df[0:2])
# also we can use this way to slice columns
print(df[['A', 'B', 'C']])

# use loc(label) to select
print(df.loc[dates[1]])
# select on a multi-axis by label
print(df.loc[:, ['A', 'B']])
print(df.loc['20180102':'20180103', ['A', 'B']])

# to get a scale value
print(df.loc[dates[0], 'A'])

# to get fast access to scalar
print(df.at[dates[0], 'A'])

# using integers index to locate elements; the end number is not included
print(df.iloc[1:3, 1:3])
print(df.iloc[:, 0:3])

# also can use iloc to get a scalar
print(df.iloc[1, 1])

# similarly we can use iat to get a faster access to a scalar
print(df.iat[1, 1])

# use a boolean value to index
print(df[df.A > 0])
print(df[df > 0])

# use isin() method to filter
df5 = df.copy()
df5['E'] = ['one', 'one', 'two', 'three', 'four', 'three', 'five']
print(df5)

print(df5[df5['E'].isin(['one', 'two'])])

# setting a new column automatically aligns the data by the indexes
s1 = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('20180101', periods=5))
df['F'] = s1
print(df)

# setting value by lable
df.at[dates[0], 'A'] = 0
print(df)
# setting value by position
df.iat[0, 0] = 1
print(df)

# setting by assign with a numpy array
df.loc[:, 'D'] = np.array([5] * len(df))
print(df)

# use len to get the rows of DataFrame
print(len(df))

# make all the positive element to negative
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

################  Missing  Data #################
print(df)
# reindexing allows you to change/add/delete the index on a specified axis
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)

# drop any rows that contains missing data
print(df1.dropna(how='any'))

#  fill missing data
print(df1.fillna(value=5))

################  stats #################
# get mean value of all the columns
print(df.mean())

# mean value of all rows
print(df.mean(1))

# use shift(n) to offset value n position next
s = pd.Series([1, 3, 5, np.nan, 6, 8, 9], index=dates).shift(3)
print(pd.Series([1, 3, 5, np.nan, 6, 8, 9], index=dates))
print(s)

# apply functions to the data
df1 = df.apply(np.cumsum)
print(df)
print(df1)

# use np.random to generate rand value
s = pd.Series(np.random.randint(0, 8, size=7))
print(s)

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())

################  Merge #################
df3 = pd.DataFrame(np.random.randn(8, 4))
print(df3)

# use concat to merge
pieces = [df3[:2], df3[2:5]]

df4 = pd.concat(pieces)
print(df4)

# also we can use pd.merge method to merge data on certain column


# join data on an key like SQL
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo1', 'foo2'], 'rval': [4, 5]})

print(left)
print(right)
print(pd.merge(left, right, on='key'))

# append rows to a dataframe
df = pd.DataFrame(np.random.randn(8, 4), columns=list('ABCD'))
print(df)

s = df.iloc[3]
print(s)

df = df.append(s, ignore_index=True)
print(df)

# time Series
rng = pd.date_range('1/1/2019',periods=100,freq='S')
ts = pd.Series(np.random.randint(0,500,len(rng)),index=rng)
pd.set_option('display.max_rows',None)
print(ts.head(105))
print(ts.resample('5Min').sum())


rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)),rng)

print(ts)

ts_utc = ts.tz_localize('UTC')
print(ts_utc)


# categoricals； convert the raw grades to a categorical type
df = pd.DataFrame({'id':[1,2,3,4,5,6],'raw_grade':['a','b','b','a','a','e']})
print(df['raw_grade'])
df['grade'] = df['raw_grade'].astype('category')
print(df['grade'])


df['grade'].cat.categories = ['very good','good','very bad']
print(df['grade'])