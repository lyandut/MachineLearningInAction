# _*_ coding: utf-8 _*_
from math import log


def createDataSet():
    dataSet = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算数据集的经验熵（香农熵）
    :param dataSet: 数据集，最后一列是分类标签
    :return shannonEnt: 香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 提取标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照所选列的特征划分数据集
    :param dataSet: 待划分数据集
    :param axis: 选中列索引
    :param value: 选中特征值
    :return:
    """
    retDataSet = []  # 创建新的列表对象，避免直接修改原数据集（引用对象）
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 这两行分片操作去掉第axis列的特征
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    todo
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(splitDataSet(myDat, 0, 1))
    print(splitDataSet(myDat, 0, 0))
