import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle as pk

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.inf)

data_path = '../data/raw/'
data_result = '../data/result/'

data_train = pd.read_csv(data_path + 'train.csv')

selected_features = data_train[
    ["Id", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "SalePrice"]]
print(selected_features.shape)
# 清洗数据
# selected_features = selected_features[selected_features["LotArea"] < 50000]
# selected_features = selected_features[selected_features["LotFrontage"] < 150]
print(selected_features.shape)
tempdata = selected_features[['MSSubClass', 'MSZoning']]

# print(tempdata)
enc = OneHotEncoder()
print(pd.DataFrame(enc.fit_transform(tempdata).toarray()).shape)

selected_features = pd.concat([selected_features, pd.DataFrame(enc.fit_transform(tempdata).toarray())], axis=1)

selected_features["OverallQual"] = selected_features["OverallQual"]/10
selected_features["OverallCond"] = selected_features["OverallCond"]/10

LotFrontage_max = np.max(selected_features["LotFrontage"])
LotFrontage_min = np.min(selected_features["LotFrontage"])
selected_features["LotFrontage_1"] = (selected_features["LotFrontage"] - LotFrontage_min)/(LotFrontage_max - LotFrontage_min)

LotArea_max = np.max(selected_features["LotArea"])
LotArea_min = np.min(selected_features["LotArea"])
selected_features["LotArea_1"] = (selected_features["LotArea"] - LotArea_min)/(LotArea_max - LotArea_min)

selected_features = selected_features.drop(columns=["MSSubClass", "MSZoning", "LotFrontage", "LotArea"])

print(selected_features)
print(selected_features.shape)

pk.dump(selected_features, file=open(data_result + 'feature.bin', 'wb'))


# plt.scatter(selected_features["LotArea"], selected_features["SalePrice"], s=20, c="b", marker='o')
# plt.show()

# print(data_train.head(5))
# print(data_train.describe())

# for i in data_train.columns:
#     print(i)
