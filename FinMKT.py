import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# some usable model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

import warnings

warnings.filterwarnings('ignore')


# os.chdir('/lqq/FinMKT/venv')

# Aiming to convert text information to number
# process the binary text(eg:Y or N)
def binary_text_process(binary_text, bk_data):
    for text in binary_text:
        feature = bk_data[text].values
        num = feature.shape[0]
        for i in range(num):
            if bk_data[text][i] == 'yes':
                bk_data[text][i] = 1
            elif bk_data[text][i] == 'no':
                bk_data[text][i] = 0
            else:
                bk_data[text][i] = -1
    return bk_data


# 针对顺序特征进行处理，顺序赋值，体现出大小关系
def order_text_process(order_text, bk_data):
    ordered_qualification = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                             "high.school", "professional.course", "university.degree", "unknown"]
    corresponding_value = list(range(1, len(ordered_qualification)))
    corresponding_value.append(-1)
    bk_data[order_text] = bk_data[order_text].replace(ordered_qualification, corresponding_value)  # 按顺序赋值代替
    bk_data[order_text].astype('float')
    return bk_data


# 针对无顺序多分类数据，用one-hot编码
def one_hot_coding(disorder_text, bk_data):
    disorder_text_coding = pd.get_dummies(bk_data[disorder_text])
    bk_data = pd.concat([bk_data, disorder_text_coding], axis=1)  # 加上one-hot编码形成的新特征
    bk_data.drop(disorder_text, axis=1, inplace=True)  # 删掉one-hot编码前的特征

    return bk_data


def data_preprocess(bk_data):
    # your code here
    # example:
    # x = pd.get_dummies(data)
    # 使用one-hot针对所有变量编码，容易丢失信息，故对特征进行分类
    # TODO
    # 特征分类,分为二分特征，顺序特征，以及无序特征
    binary_text = ['default', 'housing', 'loan']
    order_text = ['education']
    disorder_text = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
    x = binary_text_process(binary_text, bk_data)
    x = order_text_process(order_text, bk_data)
    x = one_hot_coding(disorder_text, bk_data)
    # your code end
    return x


def predict(x_train, x_test, y_train, Model):
    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    model = eval(Model)()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # your code here end

    return y_pred


def split_data(bk_data):
    y = bk_data.y
    x = bk_data.loc[:, bk_data.columns != 'y']
    x = data_preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])


if __name__ == '__main__':
    data = pd.read_csv('bank-additional-full.csv', sep=';')
    Models = ['KNeighborsClassifier', 'SVC', 'LogisticRegression', 'DecisionTreeClassifier', 'MLPClassifier',
              'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'BaggingClassifier']
    x_train, x_test, y_train, y_test = split_data(data)
    for Model in Models:
        y_pred = predict(x_train, x_test, y_train, Model)
        print(Model, ": ", end='')
        print_result(y_test, y_pred)
