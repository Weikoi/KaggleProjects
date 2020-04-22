import pickle as pk
import pandas as pd
import sys
from xgboost import plot_importance
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

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
import seaborn as sns

# corr = df.corr()
# plt.subplots(figsize=(15, 12))
# sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
# plt.show()

# 网络构造
network = models.Sequential()
network.add(layers.Dense(24, activation='relu', input_shape=(24,)))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                loss='mse', metrics=['mae'])

# 数据集准备
print(X_train)
print(X_train.values)

X_train = X_train.values.reshape((1080, 24 * 1))
# X_train = X_train.values.astype('float32')

X_test = X_test.values.reshape((121, 24 * 1))
# X_test = X_test.astype('float32')

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# from keras.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

# train
history = network.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

# test
val_mse, val_mae = network.evaluate(X_test, y_test)
print('test_acc:', val_mse)
print(network.predict(X_test))

# # %%
history_dict = history.history
print(history_dict.keys())
print(history_dict.values())
#
#
# # 模型保存
#
# network.save("./dnn_model.h5")
#
#
#
# # %%
import matplotlib.pyplot as plt
#
#
# print(history.history)
# acc = history.history['binary_accuracy']
# val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#
epochs = range(1, len(loss) + 1)
#
#
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#
# # %%
# # plt.clf()  # clear figure
# # acc_values = history_dict['binary_accuracy']
# # val_acc_values = history_dict['val_binary_accuracy']
#
# plt.plot(epochs, acc, '--', marker="D", label='Training acc')
# plt.plot(epochs, val_acc, '-', marker='o', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.ylim(0.6, 1.0)
# plt.show()
