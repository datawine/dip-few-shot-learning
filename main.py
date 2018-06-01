import tensorflow as tf
import numpy as np
from alexnet import AlexNet
import os
import time
import cv2

tf.app.flags.DEFINE_string("alexnet_classes", "./imagenet-classes.txt", "label dir")
tf.app.flags.DEFINE_boolean("use_alexnet", True, "use alexnet")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean("use_protonet", False, "use prototype network")
tf.app.flags.DEFINE_integer("protonet_shot", 3, "protonet shot")
tf.app.flags.DEFINE_integer("protonet_query", 5, "protonet query")
tf.app.flags.DEFINE_integer("protonet_test", 2, "protonet test")
tf.app.flags.DEFINE_integer("protonet_classnum", 50, "protonet class num")
tf.app.flags.DEFINE_integer("protonet_classnum", 50, "protonet class num")

FLAGS = tf.app.flags.FLAGS

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

    return np.concatenate((img_bgr, img_bgr, img_bgr), axis=0)

def readAlexnetLabel():
    with open(FLAGS.alexnet_classes, "r") as f:
        return f.readlines()    
alexnet_label = readAlexnetLabel()

with tf.Session() as sess:
    if FLAGS.use_alexnet:
        if FLAGS.use_protonet == False:
            input_layer = tf.placeholder(tf.float32, [3, 227, 227, 3])
            model = AlexNet(input_layer, FLAGS.keep_prob, 1000, [])

            pred = tf.cast(tf.argmax(model.fc8, 1), tf.int32)
            tf.global_variables_initializer().run()

            model.load_initial_weights(sess)

            input_img = genInput("./training/009.bear/009_0001.jpg")
            pred = sess.run([pred], feed_dict={input_layer: input_img})[0]
            print pred
        elif FLAGS.use_protonet == True:
            support_dim = FLAGS.protonet_classnum * FLAGS.protonet_shot
            query_dim = FLAGS.protonet_classnum * FLAGS.protonet_query

            input_support = tf.placeholder(tf.float32, [support_dim, 227, 227, 3])
            input_query = tf.placeholder(tf.float32, [query_dim, 227, 227, 3])
            y = tf.placeholder(tf.int64, [FLAGS.protonet_classnum, FLAGS.protonet_query])
            y_one_hot = tf.one_hot(y, depth = FLAGS.protonet_classnum)

            input_total = tf.concat([input_support, input_query], 0)
            model = AlexNet(input_total, FLAGS.keep_prob, 1000, [])

            def addXWB(x, num_in, num_out, name):
                with tf.variable_scope(name) as scope:
                    weights = tf.get_variable('weights', shape=[num_in, num_out],
                                            trainable=True)
                    biases = tf.get_variable('biases', [num_out], trainable=True)
                    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
                    return act
            fc9 = addXWB(model.fc8, 1000, 256, "fc9")
            fc10 = addXWB(model.fc9, 256, 50, "fc9")
            emb_support = tf.slice(fc9, [0, 0], [support_dim, 50])
            emb_query = tf.slice(fc9, [support_dim, 0], [query_dim, 50])

            def euclidean_distance(a, b):
                N, D = tf.shape(a)[0], tf.shape(a)[1]
                M = tf.shape(b)[0]
                a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
                b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
                return tf.reduce_mean(tf.square(a - b), axis=2)
            dists = euclidean_distance(emb_support, emb_query)
            log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [FLAGS.protonet_classnum, FLAGS.protonet_shot, -1])
            ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))

            acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

            tf.global_variables_initializer().run()

            model.load_initial_weights(sess)
