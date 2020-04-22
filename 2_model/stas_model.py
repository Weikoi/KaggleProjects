import pickle as pk
import pandas as pd
import sys
from xgboost import plot_importance
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

data_result = '../data/result/'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pk.load(file=open(data_result + 'feature.bin', 'rb'))
df.dropna(inplace=True)
print(np.isnan(df).any())

columns = list(df.columns)
columns.remove("Id")
columns.remove("SalePrice")

print("================== 正在加载数据集 ==================")

X = df[columns]
y = df["SalePrice"]

print("================== 正在构建数据特征 ================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# print("X_train 占用内存大小为：", round(sys.getsizeof(X_train) / 1024 / 1024, 2), "MB")


print("================== 选择模型 ======================")
"""
选择训练模型
"""
import time

time_s = time.time()
model = 'rf'

print("================== 正在构建模型 ===================")

"""
逻辑回归 单次验证：0.98635
"""
if model == 'lr':
    lg_clf = LinearRegression()
    lg_clf.fit(X_train, y_train)
    y_pred_lr = lg_clf.predict(X_test)

    print('MSE为：', mean_squared_error(y_test, y_pred_lr))
    print('MSE为(直接计算)：', np.mean((y_test - y_pred_lr) ** 2))
    print('RMSE为：', np.sqrt(mean_squared_error(y_test, y_pred_lr)))

"""
SVM 单次验证：0.99095
"""
if model == 'svm':
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    print("SVM:", accuracy_score(y_test, y_pred))

"""
tree 单次验证：0.99965
"""
if model == 'dt':
    tree_rg = tree.DecisionTreeRegressor()
    tree_rg.fit(X_train, y_train)
    y_pred_rf = tree_rg.predict(X_test)

    print('MSE为：', mean_squared_error(y_test, y_pred_rf))
    print('MSE为(直接计算)：', np.mean((y_test - y_pred_rf) ** 2))
    print('RMSE为：', np.sqrt(mean_squared_error(y_test, y_pred_rf)))

"""
GaussianNB 单次验证：0.92
"""
if model == 'gnb':
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    print("GaussianNB:", accuracy_score(y_test, y_pred))

"""
随机森林 单次验证 score 0.99355
"""
if model == 'rf':
    forest_reg = RandomForestRegressor()
    forest_reg.fit(X_train, y_train)
    y_pred_rf = forest_reg.predict(X_test)

    print('MSE为：', mean_squared_error(y_test, y_pred_rf))
    print('MSE为(直接计算)：', np.mean((y_test - y_pred_rf) ** 2))
    print('RMSE为：', np.sqrt(mean_squared_error(y_test, y_pred_rf)))

"""
随机森林 grid search 五折交叉
Best: 0.762338 using {'max_leaf_nodes': 38, 'n_estimators': 1200}
"""
if model == 'rf_gs':
    rnd_clf = RandomForestClassifier(n_jobs=-1)
    param_grid = {"max_leaf_nodes": [i for i in range(6, 40, 2)],
                  "n_estimators": [i for i in range(200, 2000, 200)]}  # 转化为字典格式，网络搜索要求

    grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)

    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print("RandomForest:", accuracy_score(y_test, y_pred_rf))

    for score in rnd_clf.feature_importances_:
        print(score)
    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    print("Test set score:{:.4f}".format(grid_search.score(X_test, y_test)))

"""
ada boost grid search 五折交叉
Best: 0.684740 using {'learning_rate': 0.1, 'n_estimators': 200}
"""
if model == 'ada_gs':
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3), algorithm="SAMME.R", n_estimators=1000, learning_rate=.7
    )

    param_grid = {"learning_rate": [i / 10 for i in range(1, 11)],
                  "n_estimators": [i for i in range(200, 1100, 100)]}  # 转化为字典格式，网络搜索要求

    grid_search = GridSearchCV(ada_clf, param_grid, cv=5, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    print("Test set score:{:.4f}".format(grid_search.score(X_test, y_test)))

"""
ada boost 单次验证
"""
if model == 'ada':
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3), algorithm="SAMME.R", n_estimators=1000, learning_rate=.7
    )

    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)

    print("ada boost:", accuracy_score(y_test, y_pred_ada))

"""
xg boost 单次验证
"""
if model == 'xgb':
    xg_model = xgb.XGBRegressor(max_depth=9, learning_rate=0.3, n_jobs=-1, n_estimators=300, silent=False,
                                objective='reg:gamma')
    xg_model.fit(X_train, y_train)
    y_pred_xg = xg_model.predict(X_test)

    print(len(y_pred_xg))

    df = pd.DataFrame(np.random.rand(38596, 2))
    df[0] = [i for i in range(1, 38597)]
    df[1] = [round(i, 4) for i in y_pred_xg]
    print(df)
    df.to_csv("submission.csv", index=False, header=False, float_format='%.4f')

    # 显示重要特征
    # plot_importance(xg_model)
    # plt.show()

"""
xg boost 五折交叉
Best: 0.953996 using {'learning_rate': 0.3, 'max_depth': 9, 'n_estimators': 300}
"""
if model == 'xgb_gs':
    xg_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_jobs=-1, n_estimators=1000, silent=False,
                                objective='reg:gamma')

    param_grid = {
        "max_depth": [i for i in range(3, 10)],
        "learning_rate": [i / 10 for i in range(1, 11, 2)],
        "n_estimators": [i for i in range(200, 1100, 100)]}  # 转化为字典格式，网络搜索要求

    grid_search = GridSearchCV(xg_model, param_grid, cv=5, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)

    # xg_model.fit(X_train, y_train)
    # y_pred_xg = xg_model.predict(X_test)
    #
    # print("xg boost:", accuracy_score(y_test, y_pred_xg))
    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    print("Test set score:{:.4f}".format(grid_search.score(X_test, y_test)))
    # 显示重要特征
    # plot_importance(model)
    # plt.show()

print("共耗时：", (time.time() - time_s) / 60, "min")
