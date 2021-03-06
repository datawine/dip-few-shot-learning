import tensorflow as tf
import cv2
import numpy as np
import os

def readTrainSet():
    cnt = 0
    train_dict = {}
    for dirs in os.listdir("./training/"):
        dir_index = dirs.split(".")[0]
        if dir_index != "":
            dir_index = int(dir_index)
        else:
            continue
        train_dict[dir_index] = {}
        for f in os.listdir("./training/" + dirs):
            f_index = int(f.split("_")[1].split(".")[0])
            train_dict[dir_index][f_index] = "./training/" + dirs + "/" + f
    return train_dict

def loadTrainSet(train_dict):
    VGG_MEAN = np.tile(np.array([103.939, 116.779, 123.68]), (227, 227, 1))
    trainset = {}
    for i in range(1, 51):
        trainset[i] = {}
        for j in range(1, 11):
            img_raw = cv2.imread(train_dict[i][j])
            img_bgr = cv2.resize(img_raw, (227, 227), interpolation=cv2.INTER_CUBIC)
            img_bgr = (img_bgr - VGG_MEAN).reshape((227, 227, 3))
            trainset[i][j] = img_bgr
    return trainset

def loadTestSet():
    testset = {}
    cnt = 0
    VGG_MEAN = np.tile(np.array([103.939, 116.779, 123.68]), (227, 227, 1))
    for fn in os.listdir("./testing"):
        cnt = cnt + 1
        img_raw = cv2.imread("./testing/" + fn)
        img_bgr = cv2.resize(img_raw, (227, 227), interpolation=cv2.INTER_CUBIC)
        img_bgr = (img_bgr - VGG_MEAN).reshape((227, 227, 3))
        number = fn.split('.')[0].split('_')[1]
        testset[int(number)] = img_bgr

        if cnt % 100 == 0:
            print ('reading {} of {}'.format(cnt, 2500))
    return testset

def loadTestNp():
    testset = np.zeros([2500, 1, 227, 227, 3])
    cnt = 0
    VGG_MEAN = np.tile(np.array([103.939, 116.779, 123.68]), (227, 227, 1))
    for fn in os.listdir("./testing"):
        cnt = cnt + 1
        index = int(fn.split("_")[1].split(".")[0]) - 1
        img_raw = cv2.imread("./testing/" + fn)
        img_bgr = cv2.resize(img_raw, (227, 227), interpolation=cv2.INTER_CUBIC)
        img_bgr = (img_bgr).reshape((227, 227, 3))
        testset[index][0] = img_bgr - VGG_MEAN

        if cnt % 100 == 0:
            print ('reading {} of {}'.format(cnt, 2500))
    return testset

def loadCorrectLabel():
    ans = {}
    with open("./data/out.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            index = int(line.split(" ")[0])
            label = int(line.split(" ")[1])
            ans[index] = label
    return ans

def vote(a,b,c):
	if a == b:
		return a
	if b == c:
		return b
	if a == c:
		return a
	return a

