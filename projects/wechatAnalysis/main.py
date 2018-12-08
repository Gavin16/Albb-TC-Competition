# -*-coding:utf-8-*-
import csv

import itchat
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from jieba.analyse import extract_tags
from snownlp import SnowNLP
from pyecharts import Map

plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图时可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 绘图时可以显示中文
import TencentYoutuyun
import pandas as pd

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image


def fun_analyse_sex(friends):
    sexs = list(map(lambda x: x['Sex'], friends[1:]))  # 收集性别数据
    counts = list(map(lambda x: x[1], Counter(sexs).items()))  # 统计不同性别的数量
    counts = sorted(counts)
    labels = ['保密', '男', '女']  # 2:女，1：男，0：保密
    colors = ['red', 'yellow', 'blue']
    plt.figure(figsize=(8, 5), dpi=80)
    plt.axes(aspect=1)
    plt.pie(counts,  # 性别统计结果
            labels=labels,  # 性别展示标签
            colors=colors,  # 饼图区域配色
            labeldistance=1.1,  # 标签距离圆点距离
            autopct='%3.1f%%',  # 饼图区域文本格式
            shadow=False,  # 饼图是否显示阴影
            startangle=90,  # 饼图起始角度
            pctdistance=0.6  # 饼图区域文本距离圆点距离
            )
    plt.legend(loc='upper left')  # 标签位置
    plt.title(u'%s的微信好友性别比例' % friends[0]['NickName'])
    plt.show()


# 签名分析
def analyseSignature(friends):
    signatures = ''
    emotions = []
    pattern = re.compile("1f\d.+")
    for friend in friends:
        signature = friend['Signature']
        if (signature != None):
            signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
            signature = re.sub(r'1f(\d.+)', '', signature)
            if (len(signature) > 0):
                nlp = SnowNLP(signature)
                emotions.append(nlp.sentiments)
                signatures += ' '.join(extract_tags(signature, 5))
    with open('signatures.txt', 'wt', encoding='utf-8') as file:
        file.write(signatures)

    # Sinature WordCloud
    back_coloring = np.array(Image.open('lvluo.jpg'))
    wordcloud = WordCloud(
        font_path='simfang.ttf',
        background_color="white",
        max_words=1200,
        mask=back_coloring,
        max_font_size=75,
        random_state=45,
        width=960,
        height=720,
        margin=15
    )

    wordcloud.generate(signatures)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('signatures.jpg')

    # Signature Emotional Judgment
    count_good = len(list(filter(lambda x: x > 0.66, emotions)))
    count_normal = len(list(filter(lambda x: x >= 0.33 and x <= 0.66, emotions)))
    count_bad = len(list(filter(lambda x: x < 0.33, emotions)))
    labels = [u'负面消极', u'中性', u'正面积极']
    values = (count_bad, count_normal, count_good)
    plt.rcParams['font.sans-serif'] = ['simHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel(u'情感判断')
    plt.ylabel(u'频数')
    plt.xticks(range(3), labels)
    plt.legend(loc='upper right', )
    plt.bar(range(3), values, color='rgb')
    plt.title(u'%s的微信好友签名信息情感分析' % friends[0]['NickName'])
    plt.show()


def get_attr(friends, key):
    return list(map(lambda user: user.get(key), friends))


def fun_pos(friends):
    users = dict(province=get_attr(friends, "Province"),
                 city=get_attr(friends, "City"),
                 nickname=get_attr(friends, "NickName"))
    provinces = pd.DataFrame(users)
    provinces_count = provinces.groupby('province', as_index=True)['province'].count().sort_values()
    attr = list(map(lambda x: x if x != '' else '未知', list(provinces_count.index)))  # 未填写地址的改为未知
    value = list(provinces_count)
    map_1 = Map("微信好友位置分布图", title_pos="center", width=1000, height=500)
    map_1.add('', attr, value, is_label_show=True, is_visualmap=True, visual_text_color='#000', visual_range=[0, 120])
    map_1


# 地理位置分析
def analyseLocation(friends):
    headers = ['NickName', 'Province', 'City']
    with open('location.csv', 'w', encoding='utf-8', newline='', ) as csvFile:
        writer = csv.DictWriter(csvFile, headers)
        writer.writeheader()
        for friend in friends[1:]:
            row = {}
            row['NickName'] = friend['NickName']
            row['Province'] = friend['Province']
            row['City'] = friend['City']
            writer.writerow(row)


if __name__ == "__main__":
    itchat.auto_login(hotReload=True)
    friends = itchat.get_friends(update=True)
    # analyseSignature(friends)
    fun_pos(friends)
    # analyseLocation(friends)
