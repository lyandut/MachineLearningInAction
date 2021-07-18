# _*_ coding: utf-8 _*_
from numpy import *
from os import listdir
from kNN.kNN101.kNN101 import classify0


def img2vector(filename):
    """
    32*32的二进制图像转换成1*1024的向量。
    :param filename:
    :return returnVect: 转换后的向量
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    # 训练集
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  # 特征矩阵，m*1024
    hwLabels = []  # 标签向量，m*1
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉文件扩展名
        classNumStr = int(fileStr.split('_')[0])  # 分类数字，即标签
        trainingMat[i] = img2vector("trainingDigits/%s" % fileNameStr)
        hwLabels.append(classNumStr)
    # 测试集
    testFileList = listdir("testDigits")
    mTest = len(testFileList)
    errorCount = 0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)  # 测试向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / mTest))


if __name__ == '__main__':
    handwritingClassTest()
