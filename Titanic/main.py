# -*- coding: utf-8 -*-

# *************************************************************
# Copyright (c) Huang Zengrui
#
# Author: Huang Zengrui <huangzengrui@yahoo.com>
# CreateTime: 2022/1/5 14:56
# DESC: 
# *************************************************************

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

print(os.getcwd())
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# %%
df_train = pd.read_csv('./data/train.csv')
df_submission = pd.read_csv('./data/test.csv')

df = df_train.append(df_submission)

# EDA
print(df.info())
print(df.describe())
print(df["Pclass"].unique())
print(df["Sex"].unique())

df["Age"].fillna(df["Age"].mean(), inplace=True)
df['AgeBand'] = pd.cut(df['Age'], [0, 5, 10, 15, 20, 25, 30, 50, 80], labels=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1],
                       include_lowest=True)

df["Fare"].fillna(df["Fare"].median(), inplace=True)
df['FareBand'] = pd.cut(df['Fare'], [0, 5, 15, 30, 200, 1000], labels=[0.2, 0.4, 0.6, 0.8, 1], include_lowest=True)

df['Has_family'] = df['SibSp'] + df['Parch']
# df['SibSpBand'] = pd.cut(df['SibSp'], [0, 1, 2, 3, 4, 6, 100], labels=[0, 1], include_lowest=True)
#
# df['ParchBand'] = pd.cut(df['Parch'], [0, 1, 100], labels=[0, 1], include_lowest=True)
df['Has_family_Band'] = pd.cut(df['Has_family'], [0, 1, 2, 3, 4, 5, 100], labels=[0, 0.2, 0.4, 0.6, 0.8, 1],
                               include_lowest=True)

df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

df["Pclass"] = df["Pclass"].map({1: 1, 2: 0.66, 3: 0.33})

# print(X["FareBand"].unique())
df_train_test = df[:891]
df_submission = df[891:]

y = df_train_test['Survived']
X = df_train_test.drop(
    ["Name", "Ticket", "Cabin", "Embarked", "Age", "Fare", "SibSp", "Parch", "Has_family", "Survived", "PassengerId"],
    axis=1)

X_submission = df_submission.drop(
    ["Name", "Ticket", "Cabin", "Embarked", "Age", "Fare", "SibSp", "Parch", "Has_family"],
    axis=1)
print(X)
print(y)

# print(np.isnan(X).any())
# print(np.isnan(y).any())
# print(X.FareBand.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
以下為硬投票方式
hard voting
"""
# %%
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score

print("==========================================")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
以下為軟投票方式
soft voting
"""
# %%
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)], voting='soft')
voting_clf.fit(X_train, y_train)

"""
分别fit各个单独分类器与集成分类器做对比
"""
# %%
from sklearn.metrics import accuracy_score

print("==========================================")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

"""
以下是Bagging集成方式,将oob_score设为True来计算oob_score()的值
ensemble bagging 
"""
# %%
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("==========================================")
print("bagging:", accuracy_score(y_test, y_pred))
print("oob_score:", bag_clf.oob_score_)

# %%
"""
以下是Pasting集成方式, 注意Pasting方式不能测算oob_score的值
ensemble pasting (No oob score)
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=False, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("==========================================")
print("pasting:", accuracy_score(y_test, y_pred))
# print(bag_clf.oob_score_)


# %%
"""
以下是随机森林方式
RandomForest
"""
from sklearn.ensemble import RandomForestClassifier

rnd_clf_ = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf_.fit(X_train, y_train)
y_pred_rf = rnd_clf_.predict(X_test)
print("==========================================")
print("RandomForest:", accuracy_score(y_test, y_pred_rf))

"""
lightGBM
"""
# import lightgbm as lgb
#
# X, val_X, y, val_y = train_test_split(
#     X_train,
#     y_train,
#     test_size=0.05,
#     random_state=1,
#     stratify=y_train  ## 这里保证分割后y的比例分布与原数据一致
# )
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# # specify your configurations as a dict
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 2,
#     'metric': 'multi_error',
#     'num_leaves': 300,
#     'min_data_in_leaf': 100,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'lambda_l1': 0.4,
#     'lambda_l2': 0.5,
#     'min_gain_to_split': 0.2,
#     'verbose': 5,
#     'is_unbalance': True
# }
#
# # train
# print('Start training...')
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10000,
#                 valid_sets=lgb_eval,
#                 early_stopping_rounds=500)
#
# print('Start predicting...')
#
# preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果
# %%
"""
特征评分
feature importance
"""

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
print("==========================================")
for name, score in zip(X_train.columns, rnd_clf.feature_importances_):
    print(name, score)
print(pd.DataFrame(index=X_train.columns, data=rnd_clf.feature_importances_, columns=['Feature Importance']))

"""
submission
"""

# print(X_submission)

ps_id = X_submission['PassengerId']
X_submission = X_submission.drop(['PassengerId', 'Survived'], axis=1)
print(X_submission.isna().any())
print(X_submission.isna().sum())
print(X.shape)
y_submission = rnd_clf_.predict(X_submission)

df_sub = pd.DataFrame()
df_sub["PassengerId"] = ps_id
df_sub["Survived"] = y_submission.astype(int)

print(df_sub)
df_sub.to_csv("./data/submission.csv", index=None)
