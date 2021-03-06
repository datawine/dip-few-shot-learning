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

def getFc7Array(mean=False, normalization=False, reorder=False):
    fc7 = np.load('./fc7.npy')
    if mean:
        vector_num = {}
        for i in range(0, 1000):
            vector_num[i] = 0
        labels = np.load('./label_correct.npy')
        print(len(labels))
        mean_fc7 = np.zeros((1000, len(fc7[0])))
        for i in range(len(fc7)):
            mean_fc7[int(labels[i])] += fc7[i]
            vector_num[labels[i]] += 1
        for i in range(0, 1000):
            mean_fc7[i] /= vector_num[i]
        if normalization:
            for i in range(0, 1000):
                mean_fc7[i] = normalize(mean_fc7[i])
        return mean_fc7
    elif reorder:
        labels = np.load('./label.npy')
        reorder_fc7 = np.zeros([len(fc7), len(fc7[0])])
        place = 0
        for i in range(0, 1000):
            for j in range(0, len(fc7)):
                if labels[j] == i:
                    reorder_fc7[place] = fc7[j]
                    place += 1
        return np.array(reorder_fc7)
    else:
        return np.array(fc7)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def getTestLabel():
    f = open('./out.txt')
    data = f.read().split('\n')
    ans = []
    dicts = {}
    for i in data:
        array = i.split(' ')
        if len(array) != 2:
            continue
        dicts[array[0]] = array[1]
    for i in range(1, len(dicts)+1):
        ans.append(dicts[str(i)])
    return ans