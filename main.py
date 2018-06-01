import tensorflow as tf
import numpy as np
from alexnet import AlexNet
import os
import time
import cv2

tf.app.flags.DEFINE_string("alexnet_classes", "./imagenet-classes.txt", "label dir")
tf.app.flags.DEFINE_boolean("use_alexnet", True, "use alexnet")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
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
    
    img_bgr = img_bgr - VGG_MEAN

    return img_bgr.reshape((1, 227, 227, 3))

def readAlexnetLabel():
    with open(FLAGS.alexnet_classes, "r") as f:
        return f.readlines()    
alexnet_label = readAlexnetLabel()

with tf.Session() as sess:
    input_layer = tf.placeholder(tf.float32, [1, 227, 227, 3])

    if FLAGS.use_alexnet:

        img = cv2.imread("./training/009.bear/009_0001.jpg")
        

        model = AlexNet(input_layer, FLAGS.keep_prob, 1000, [])
        sm = tf.nn.softmax(model.fc8)
        pred = tf.cast(tf.argmax(sm, 1), tf.int32)

        tf.global_variables_initializer().run()

        model.load_initial_weights(sess)

        input_img = genInput("./training/009.bear/009_0001.jpg")
        pred = sess.run([pred], feed_dict={input_layer: input_img})
        print pred
#        print alexnet_label[np.array(pred)[0]]