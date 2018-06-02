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
            img_bgr = (img_bgr - VGG_MEAN).reshape((1, 227, 227, 3))
            trainset[i][j] = img_bgr
    return trainset