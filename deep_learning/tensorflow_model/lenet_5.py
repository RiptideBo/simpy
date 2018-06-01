'''
使用tensorflow复现LeNet-5模型，用于Mnist数据

    - 学习tf教程中的模型写法
    - 学习tensorboard的用法

备注：
    目前只是构筑了模型，数据的训练和测试部分没有。

python: 3.5.2
author: Stephen Lee
github : https://github.com/RiptideBo
2017.12.20

'''

import tensorflow as tf
import numpy as np


def inference(placehold_input):
    '''
    模型部分 - graph
    '''

    def weight_variable(shape, **kwargs):
        return tf.Variable(tf.truncated_normal(shape=shape,
                                               stddev=0.1,
                                               dtype=tf.float32), **kwargs)

    def bias_variable(shape, **kwargs):
        return tf.Variable(tf.constant(0., shape=shape, dtype=tf.float32), **kwargs)


    with tf.name_scope('Conv_1'):
        weights = weight_variable(shape=[5,5,1,6],name='weights')
        bias = bias_variable(shape=[6],name='bias')
        conv1 = tf.nn.conv2d(placehold_input,
                             weights,
                             strides=[1,1,1,1],
                             padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1,bias),name='conv1')

        print(conv1.get_shape().as_list())

    with tf.name_scope('MaxPooling_1'):
        pooling_1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],
                       padding='VALID',name='maxpool_1')

        print(pooling_1.get_shape().as_list())

    with tf.name_scope('Conv_2'):
        weights = weight_variable(shape=[5,5,6,16],name='weights')
        bias = bias_variable(shape=[16],name='bias')
        conv2 = tf.nn.conv2d(pooling_1,
                             weights,
                             strides=[1,1,1,1],
                             padding='VALID')

        conv2 = tf.nn.relu(tf.nn.bias_add(conv2,bias),name='conv2')
        print(conv2.get_shape().as_list())

    with tf.name_scope('MaxPooling_2'):
        pooling_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='VALID', name='maxpool_2')
        print(pooling_2.get_shape().as_list())

    # reshape (flatten)
    with tf.name_scope('Flatten'):
        shape_pooling2 = pooling_2.get_shape().as_list()
        flatten_units = np.cumproduct(shape_pooling2[1:])[-1]
        flatten = tf.reshape(pooling_2,
                             shape=(-1, flatten_units),
                             name='flatten')
        print(flatten.get_shape().as_list())

    with tf.name_scope('Hiden_1'):
        weights = weight_variable(shape=[flatten_units,120],name='weights')
        bias = bias_variable(shape=[120], name='bias')

        wx_plus_b = tf.matmul(flatten,weights) + bias
        hiden1 = tf.nn.relu(wx_plus_b,name='hiden1')
        print(hiden1.get_shape().as_list())

    with tf.name_scope('Hiden_2'):
        weights = weight_variable(shape=[120, 84], name='weights')
        bias = bias_variable(shape=[84], name='bias')
        wx_plus_b = tf.matmul(hiden1, weights) + bias
        hiden2 = tf.nn.relu(wx_plus_b, name='hiden2')
        print(hiden2.get_shape().as_list())

    with tf.name_scope('Output'):
        weights = weight_variable(shape=[84,10],name='weights')
        bias = bias_variable(shape=[10], name='bias')

        logit = tf.add(tf.matmul(hiden2, weights),bias,name='logit')
        print(logit.get_shape().as_list())

    return logit


def training(label_placeholder,logit,learning_rate):
    '''
    identify loss and train op
    '''
    with tf.name_scope('Loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder,
                                                logits=logit,
                                                dim=-1)
        loss = tf.reduce_mean(loss,name='loss')

        tf.summary.scalar('loss', loss)
    with tf.name_scope('Training'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return loss,train_op


placeholder_x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name='input')
placeholder_y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='labels')


with tf.Session() as sess:

    logit = inference(placeholder_x)
    loss,train_op = training(placeholder_y, logit, learning_rate=.01)

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(r'G:\data\cache\lenet', sess.graph)

    sess.run(tf.global_variables_initializer())










