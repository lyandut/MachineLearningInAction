# _*_ coding: utf-8 _*_
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from models.kNN101.kNN101 import classify0


def file2matrix(filename):
    """
    :param filename:
    :return returnMat: 特征矩阵
    :return classLabelVector: 标签向量
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))  # 创建以0填充的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 标签按整型存储
        if listFromLine[-1] == "didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1] == "largeDoses":
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


def showData(datingDataMat, datingLabels):
    """
    训练样本集可视化
    :param datingDataMat: 特征矩阵
    :param datingLabels: 标签向量
    :return:
    """
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 需要查看自己的电脑是否会包含该字体
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', fontproperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', fontproperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', fontproperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', fontproperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', fontproperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


def autoNorm(dataSet):
    """
    将数字特征值归一化到[0,1]区间
    :param dataSet: 特征矩阵
    :return normDataSet: 归一化后的特征矩阵
    :return ranges: 数据范围
    :return minVals: 最小值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(ranges, (m, 1))  # (原始值-最小值)/(最大值-最小值)
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    测试分类器，使用错误率评价分类器效果
    :return:
    """
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    hoRatio = 0.10
    numTestVecs = int(m * hoRatio)  # 取10%的数据作为测试集
    errorCount = 0  # 预测错误计数
    for i in range(numTestVecs):
        # [0, numTestVecs-1]是测试集，[numTestVecs, m-1]是训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (float(errorCount) / float(numTestVecs)))


def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    # 用户输入三维特征值
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


if __name__ == '__main__':
    datingClassTest()
    # classifyPerson()
