# _*_ coding: utf-8 _*_
from numpy import *
import operator


def createDataSet():
    """
    创建数据集和标签
    :return group: 数据集
    :return labels: 标签
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k-邻近算法
    :param inX: 用于分类的输入向量（测试集）
    :param dataSet: 训练样本集
    :param labels: 标签
    :param k: kNN参数
    :return sortedClassCount[0][0]: 统计次数最多的类别
    """
    # dataSetSize = dataSet.shape[0]  # dataSet行数
    # diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 将inX在行方向上复制dataSetSize次，做减法
    # sqDiffMat = diffMat ** 2  # array的*是矩阵数量积运算，对应位置元素相乘
    # sqDistances = sqDiffMat.sum(axis=1)  # 按行相加
    # distances = sqDistances ** 0.5  # 至此，欧式距离计算完毕

    distances = sum((inX - dataSet) ** 2, axis=1) ** 0.5  # 一行计算欧氏距离

    sortedDistIndicies = distances.argsort()  # 距离从小到大排序，排序对象是索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 第i个元素的类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计类别次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按值降序排序
    return sortedClassCount[0][0]  # 返回次数最多的类别


if __name__ == '__main__':
    group, labels = createDataSet()
    test = array([0, 0])
    test_label = classify0(test, group, labels, 3)
    print(test_label)
