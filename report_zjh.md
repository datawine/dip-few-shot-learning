# 数图大作业报告

### 计52 周京汉 2015011245

## 问题概述

本次大作业的具体任务为：在一个已经训练好了的1000类的网络，及其每个类别几十个样本所输出的fc7层的输出和其对应的labels等数据已知的情况下，给50个新的类别，每个类别给10张图片用来训练，最终得出对于新的类别的区分的网络。

这个问题是典型的few-shot问题，由少量的训练集得出正确率比较高的预测模型。我们在阅读文献之后，尝试了各种基础做法与尝试。我在其中主要的任务是完成分类器类型的方法的研究与代码书写调参等。

## 文献阅读

在小组分工当中，我阅读的是第二篇文献，《Learning to Compare: Relation Network for Few-Shot Learning》这是一篇应用了`relation network`的思想来进行few-shot任务完成的。其最主要的思想如下图：

![](/Users/mac/Desktop/university/CST/1718Spring/6_数字图像处理/hw/2/dip-few-shot-learning/report/zjh_1.jpeg)

其中，左边5幅图是用于训练的训练集，底下那张是用来测试的。在训练之后，会生成一张网络，即为中间的“embedding module”

## 创新思路

## 个人贡献

