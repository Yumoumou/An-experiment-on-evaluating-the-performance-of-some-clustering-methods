import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


# 读取数据函数,输入为数据文件名和训练、测试切分比率，返回为list类型的训练数据集和测试数据集
def loadData(fileName_data, fileName_label):
    dataset = []
    labelset = []  # 标签
    with open(fileName_label) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split(',')  # 去除空白和逗号“,”
            labelset.append(lineData)
        labelset = np.array(labelset)
    with open(fileName_data) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split()  # 去除空白和逗号“,”
            dataset.append(lineData)
        dataset = np.array(dataset)

    return dataset, labelset


# 计算两个点之间的欧拉距离
def euclidean_distance(X1, X2, feature_num):
    distance = 0.0
    for x in range(feature_num):
        distance += pow((X1[x] - X2[x]), 2)
    return distance


def plot_scatter(data, label, colValue):
    X2 = np.c_[data, label]
    for point in X2:
        colorIndex = int(point[-1])  # 样本点的聚类标签
        x = point[0]
        y = point[1]
        plt.scatter(x, y, marker='o', color=colValue[colorIndex - 1], label=colorIndex)


def preprocess(y):
    new = []
    for y1 in y:
        new.append(y1[0])
    new = np.array(new)
    return new


# 读入数据集2
fileName_data = 'ls3.txt'
fileName_label = 'ls3_cl.txt'
X, y = loadData(fileName_data, fileName_label)
X = X.astype('int64')
y = y.astype('int64')
y = preprocess(y)
num0fclusters = len(set(y))
print("num of clusters:", len(set(y)))
# 数据可视化
colValue = ['r', 'y', 'g', 'b', 'c', 'm']  # 创建颜色数组 r代表红色 y代表黄色 g代表绿色
# plt.figure()
# plot_scatter(X, y, colValue)
# plt.title("expected results")
# plt.show()

# 调用库函数进行K-means聚类
# k = num0fclusters
# cluster = KMeans(n_clusters=k, random_state=0).fit(X)  # 进行k-means聚类
# centroid = cluster.cluster_centers_  # 查看簇心
# # print("centroid:",centroid)
# pre_labels = cluster.labels_  # 聚类后的预测标签
# # 聚类结果可视化(
# plt.figure()
# plot_scatter(X, pre_labels, colValue)
# plt.title("K-means predicted results, k=6")
# plt.show()
# # 性能衡量
# ARI = metrics.adjusted_rand_score(y, pre_labels)
# print('K-means ARI: %.2f' % ARI)
#
#
# # 调用库函数进行DBSCAN聚类
# cluster = DBSCAN(eps=10, min_samples=5).fit(X)
# pre_labels = cluster.labels_
# num0fclusters = len(set(pre_labels)) - (1 if -1 in pre_labels else 0)
# print("predicted number of predicted:", num0fclusters)
# plt.figure()
# plot_scatter(X, pre_labels, colValue)
# plt.title("DBSCAN predicted results")
# plt.show()
# # 性能衡量
# ARI = metrics.adjusted_rand_score(y, pre_labels)
# print('DBSCAN ARI: %.2f' % ARI)

# 层次聚类
ac = AgglomerativeClustering(n_clusters=num0fclusters, affinity='euclidean', linkage='single')
ac.fit(X)
pre_labels = ac.labels_
plt.figure()
plot_scatter(X, pre_labels, colValue)
plt.title("AGNE predicted results")
plt.show()
# 性能衡量
ARI = metrics.adjusted_rand_score(y, pre_labels)
print('AGNE ARI: %.2f' % ARI)
