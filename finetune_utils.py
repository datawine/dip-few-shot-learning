import numpy as np
from utils import *

item_num = 8
class_num = 50
test_num = 2

def genTrainSet(train_set, num):

    x = np.zeros([num, 227, 227, 3])
    y = np.zeros([num])

    select_list = np.random.permutation(class_num * item_num)[:num]
    for i, select_item in enumerate(select_list):
        class_index = select_item / item_num
        class_index = int(class_index)
        item_index = select_item % item_num
        x[i] = train_set[class_index + 1][item_index + 1]
        y[i] = class_index

    return x, y

def genTestSet(train_set):
    num = 100

    x = np.zeros([100, 227, 227, 3])
    y = np.zeros([100])

    for i in range(50):
        x[i * 2] = train_set[i + 1][9]
        x[i * 2 + 1] = train_set[i + 1][10]
        y[i * 2] = i
        y[i * 2 + 1] = i

    return x, y

def getFc7Array():
    fc7 = np.load('./fc7.npy')
    return np.array(fc7)
