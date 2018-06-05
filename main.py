# -*- coding:utf-8 -*-
import xgboost as xgb
import tensorflow as tf
import numpy as np
from alexnet import AlexNet
import os
import time
import cv2
from utils import *
import finetune_utils
<<<<<<< HEAD
from sklearn import neighbors
import sys
=======
from sklearn import neighbors  
from sklearn.svm import SVC, LinearSVC
>>>>>>> 1b06ad804087b516ff25d9b27170144562b8f48a

tf.app.flags.DEFINE_string("alexnet_classes", "./imagenet-classes.txt", "label dir")
tf.app.flags.DEFINE_boolean("use_alexnet", True, "use alexnet")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")

tf.app.flags.DEFINE_boolean("use_raw_alexnet", False, "use prototype network")

tf.app.flags.DEFINE_boolean("use_protonet", True, "use prototype network")
tf.app.flags.DEFINE_integer("protonet_selected", 8, "protonet select num")
tf.app.flags.DEFINE_integer("protonet_shot", 5, "protonet shot")
tf.app.flags.DEFINE_integer("protonet_query", 2, "protonet query")
tf.app.flags.DEFINE_integer("protonet_classnum", 50, "protonet class num")
tf.app.flags.DEFINE_integer("protonet_epoch", 50, "protonet train epoch")

tf.app.flags.DEFINE_boolean("use_finetune_1", False, "finetune 1")
tf.app.flags.DEFINE_integer("finetune1_classnum", 50, "protonet class num")
tf.app.flags.DEFINE_integer("finetune1_itemnum", 10, "protonet class num")
tf.app.flags.DEFINE_integer("finetune1_epochnum", 100, "protonet class num")
tf.app.flags.DEFINE_integer("finetune1_epoch", 150, "protonet train epoch")

tf.app.flags.DEFINE_boolean("use_finetune_dot", False, "finetune dot")
tf.app.flags.DEFINE_integer("train_class_num", 50, "train class num")
tf.app.flags.DEFINE_integer("train_pic_num", 10, "train pic num")

tf.app.flags.DEFINE_boolean("use_xgboost", True, "use xgboost")

FLAGS = tf.app.flags.FLAGS

xgb_param = {}
xgb_param['objective'] = 'multi:softmax'
#xgb_param['eta'] = 0.2
#xgb_param['max_depth'] = 10
xgb_param['silent'] = 1
xgb_param['num_class'] = 50

def genInput(filename):
    '''
    VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
 	# load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, VGG_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    img_bgr = tf.reshape(img_bgr, [1, 227, 227, 3])
    '''

    VGG_MEAN = np.tile(np.array([103.939, 116.779, 123.68]), (227, 227, 1))
    img_raw = cv2.imread("./training/009.bear/009_0001.jpg")

    img_bgr = cv2.resize(img_raw, (227, 227), interpolation=cv2.INTER_CUBIC)
    
    img_bgr = (img_bgr - VGG_MEAN).reshape((1, 227, 227, 3))

    return img_bgr

def readAlexnetLabel():
    with open(FLAGS.alexnet_classes, "r") as f:
        return f.readlines()    
alexnet_label = readAlexnetLabel()

def readFewshotLabel():
    fewshot_label = []
    for dirs in os.listdir("./training"):
        fewshot_label.append(dirs.split(".")[1])
    return fewshot_label
fewshot_label = readFewshotLabel()

def addXWB(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    return act

with tf.Session() as sess:
    if FLAGS.use_alexnet:
        if FLAGS.use_raw_alexnet == True:
            input_layer = tf.placeholder(tf.float32, [None, 227, 227, 3])
            model = AlexNet(input_layer, FLAGS.keep_prob, 1000, [])

            pred = tf.cast(tf.argmax(model.fc8, 1), tf.int32)
            tf.global_variables_initializer().run()

            model.load_initial_weights(sess)

            input_img = genInput("./training/009.bear/009_0001.jpg")
            pred = sess.run([pred], feed_dict={input_layer: input_img})[0]
            print (pred)
        elif FLAGS.use_protonet == True:
            input_support = tf.placeholder(tf.float32, [None, None, 227, 227, 3])
            input_query = tf.placeholder(tf.float32, [None, None, 227, 227, 3])
            support_shape = tf.shape(input_support)
            query_shape = tf.shape(input_query)
            num_classes, num_support = support_shape[0], support_shape[1]
            num_classes2, num_queries = query_shape[0], query_shape[1]
            y = tf.placeholder(tf.int64, [None, None])
            y_one_hot = tf.one_hot(y, depth=num_classes)

            emb_input = tf.concat([tf.reshape(input_support, [num_classes * num_support, 227, 227, 3]), \
                            tf.reshape(input_query, [num_classes2 * num_queries, 227, 227, 3])], 0)            

            model = AlexNet(emb_input, 1, 1000, [])
            fc9 = addXWB(model.fc8, 1000, 512, "fc9")
            emb_dim = tf.shape(fc9)[-1]

            emb_support = tf.slice(fc9, [0, 0], [num_classes * num_support, emb_dim])
            emb_query = tf.slice(fc9, [num_classes * num_support, 0], [num_classes2 * num_queries, emb_dim])

            emb_support = tf.reduce_mean(tf.reshape(emb_support, [num_classes, num_support, emb_dim]), axis=1)

            def euclidean_distance(a, b):
                # a.shape = N x D
                # b.shape = M x D
                N, D = tf.shape(a)[0], tf.shape(a)[1]
                M = tf.shape(b)[0]
                a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
                b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
                return tf.reduce_mean(tf.square(a - b), axis=2)

            dists = euclidean_distance(emb_query, emb_support)
            log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes2, num_queries, -1])
            ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
            acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

            pred = tf.argmax(log_p_y, axis=-1)

            train_op = tf.train.AdamOptimizer().minimize(ce_loss)

            tf.global_variables_initializer().run()
            model.load_initial_weights(sess)

            train_dict = readTrainSet()
            train_set = loadTrainSet(train_dict)
            test_set = loadTestSet()

            ## train part
            support = np.zeros([FLAGS.protonet_classnum, FLAGS.protonet_shot, 227, 227, 3])
            query = np.zeros([FLAGS.protonet_classnum, FLAGS.protonet_query, 227, 227, 3])
            labels = np.zeros([FLAGS.protonet_classnum, FLAGS.protonet_query])
            for epi in range(FLAGS.protonet_epoch):
                for epi_cls in range(FLAGS.protonet_classnum):
                    selected = np.random.permutation(FLAGS.protonet_selected)[: FLAGS.protonet_shot + FLAGS.protonet_query]
                    for j, sel in enumerate(selected):
                        if j < FLAGS.protonet_shot:
                            support[epi_cls][j] = train_set[epi_cls + 1][sel + 1]
                        else:
                            query[epi_cls][j - FLAGS.protonet_shot] = train_set[epi_cls + 1][sel + 1]
                labels = np.tile(np.arange(FLAGS.protonet_classnum)[:, np.newaxis], (1, FLAGS.protonet_query)).astype(np.uint8)
                _, loss, ac, p = sess.run([train_op, ce_loss, acc, pred], feed_dict={input_support: support, input_query: query, y: labels})

                if (epi + 1) % 5 == 0:
                    print ('[episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, FLAGS.protonet_epoch, loss, ac))
            ## test part
            support = np.zeros([FLAGS.protonet_classnum, 10, 227, 227, 3])
            query = np.zeros([2500, 1, 227, 227, 3])
            for epi_cls in range(FLAGS.protonet_classnum):
                for j in range(10):
                    support[epi_cls][j] = train_set[epi_cls + 1][sel + 1]
            
            for i in range(2500):
                query[i][0] = test_set[i + 1]
            p = sess.run([pred], feed_dict={input_support: support, input_query: query})[0]

            VGG_MEAN = np.tile(np.array([103.939, 116.779, 123.68]), (227, 227, 1))
            for i in range(2500):
                print (fewshot_label[int(p[i])]) 
                cv2.imshow("img", query[i][0] + VGG_MEAN)
                cv2.waitKey(0)

        elif FLAGS.use_finetune_1 == True:
            input_x = tf.placeholder(tf.float32, [None, 227, 227, 3])
            label = tf.placeholder(tf.int32, [None])

            model = AlexNet(input_x, FLAGS.keep_prob, 1000, [])

            fc9 = addXWB(model.fc8, 1000, 256, "fc9")
            fc10 = addXWB(fc9, 256, 50, "fc10")

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=fc10))
        
            correct_pred = tf.equal(tf.cast(tf.argmax(fc10, 1), tf.int32), tf.cast(label, tf.int32))
        
            pred = tf.argmax(fc10, 1)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

            tf.global_variables_initializer().run()
            model.load_initial_weights(sess)

            train_dict = readTrainSet()
            train_set = loadTrainSet(train_dict)

            for i in range(FLAGS.finetune1_epoch):
                x, _label = finetune_utils.genTrainSet(train_set, FLAGS.finetune1_epochnum)
                _, lss, ac = sess.run([train_op, loss, acc], feed_dict={input_x: x, label: _label})

                if (i + 1) % 5 == 0:
                    print ('[episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(i + 1, FLAGS.finetune1_epoch, lss, ac))
                    
            x, _label = finetune_utils.genTestSet(train_set)
            lss, ac = sess.run([loss, acc], feed_dict={input_x: x, label: _label})
            print ("final acc =>", ac)
        elif FLAGS.use_finetune_dot == True:
            input_x = tf.placeholder(tf.float32, [None, 227, 227, 3])
            label = tf.placeholder(tf.int32, [None])
            knn = neighbors.KNeighborsClassifier(n_neighbors=8,weights='uniform')

            model = AlexNet(input_x, 1.0, 1000, [])
            data = model.fc7
            tf.global_variables_initializer().run()
            model.load_initial_weights(sess)
            fc7_vector = finetune_utils.getFc7Array()

            datas = []
            labels = []
            test_datas = []
            test_labels = []
            x = np.zeros([1, 227, 227, 3])
            y = np.zeros([1])

            train_dict = readTrainSet()
            train_set = loadTrainSet(train_dict)
            
            for i in range(1, FLAGS.train_class_num+1):
                for j in range(1, FLAGS.train_pic_num-1):
                    x = train_set[i][j]
                    y[0] = i
                    data_fc7 = sess.run([data], feed_dict={input_x: x, label: y})
                    data_vector = np.array(data_fc7).reshape((4096,1)).tolist()
                    data_for_knn = (fc7_vector.dot(data_vector)).reshape((63996)).tolist()
                    datas.append(data_for_knn)
                    labels.append(i)

                for j in range(FLAGS.train_pic_num-1, FLAGS.train_pic_num+1):
                    x = train_set[i][j]
                    y[0] = i
                    data_fc7 = sess.run([data], feed_dict={input_x: x, label: y})
                    data_vector = np.array(data_fc7).reshape((4096,1)).tolist()
                    data_for_knn = (fc7_vector.dot(data_vector)).reshape((63996)).tolist()
                    test_datas.append(data_for_knn)
                    test_labels.append(i)
                if i % 10 == 0:
                    knn.fit(datas, labels)#这个感觉好慢啊，每次都从新fit一遍
                    acc = knn.score(test_datas, test_labels)
                    print ('[class {}/{}] => acc: {:.5f}'.format(i, 50, acc))
        else: #svm
            input_x = tf.placeholder(tf.float32, [None, 227, 227, 3])
            label = tf.placeholder(tf.int32, [None])
            clf = SVC(C=10,cache_size=500,tol=1e-4,kernel='linear')
            linearSVC_clf = LinearSVC(C=10)

            model = AlexNet(input_x, 1.0, 1000, [])
            data = model.fc7
            tf.global_variables_initializer().run()
            model.load_initial_weights(sess)
            fc7_vector = finetune_utils.getFc7Array()

            datas = []
            labels = []
            test_datas = []
            test_labels = []
            x = np.zeros([1, 227, 227, 3])
            y = np.zeros([1])
                    
            train_dict = readTrainSet()
            train_set = loadTrainSet(train_dict)
            for i in range(1, FLAGS.train_class_num+1):
                for j in range(1, FLAGS.train_pic_num-1):
                    x = train_set[i][j]
                    y[0] = i
                    data_fc7 = sess.run([data], feed_dict={input_x: x, label: y})
                    data_vector = np.array(data_fc7).reshape((4096)).tolist()
                    #data_for_knn = (fc7_vector.dot(data_vector)).reshape((63996)).tolist()
                    datas.append(data_vector)
                    labels.append(i)
                
                for j in range(FLAGS.train_pic_num-1, FLAGS.train_pic_num+1):
                    x = train_set[i][j]
                    y[0] = i
                    data_fc7 = sess.run([data], feed_dict={input_x: x, label: y})
                    data_vector = np.array(data_fc7).reshape((4096)).tolist()
                    #data_for_knn = (fc7_vector.dot(data_vector)).reshape((63996)).tolist()
                    test_datas.append(data_vector)
                    test_labels.append(i)
                if i % 10 == 0:
                    clf.fit(datas, labels)
                    linearSVC_clf.fit(datas, labels)
                    acc1 = clf.score(test_datas, test_labels)
                    acc2 = linearSVC_clf.score(test_datas, test_labels)
                    print ('   SVM:     [class {}/{}] => acc: {:.5f}'.format(i, 50, acc1))
                    print ('linear_SVM: [class {}/{}] => acc: {:.5f}'.format(i, 50, acc2))


                            cnt = 0
                            for index, p in enumerate(predict):
                                if abs(p - xgb_test_label[index]) < 0.1:
                                    cnt = cnt + 1

                            print('eta: {}, max_depth: {}, num_round: {}, acc: {}'.format(eta * 0.01, max_depth, num_round, cnt / 100.0))
            '''