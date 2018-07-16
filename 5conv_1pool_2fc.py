'''Author Nyla Worker

Using code from Rankyung Hong UMN

Here I attempt to recreate AlexNet in a not distributed manner.

'''

import sys
import tensorflow as tf
from tflearn.data_utils import to_categorical
from tflearn.datasets import cifar10
from pylab import *
import numpy as np
import time


def weight_variable(shape):
    '''
    Creates a matrix
    :param shape: integer that specifies the output tensor
    :return: a tensor of random numbers with the specified shape.
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    ## tf.truncated_normal outputs a matrix of the desired shape with random variable truncated from a normal distribution

    return tf.Variable(initial)


def bias_variable(shape):
    '''

    :param shape: This is the shape of the bias matrix
    :return:
    '''

    '''
    Question why 0.1? Can't this be anything?
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=[1, 1, 1, 1], pad='SAME'):
    ''' first [1,4,4,1]

    :param x: features
    :param W: weights for the covolutional layer
    :param stride: Regulates the stride in the convolutional layer
    :return:
    '''
    '''
    Question: I don't understand how padding here works. In AlexNet there is padding = 3 of zeroes, rihgt?
    Strides: Can you confirm that the strides are corect? I am confused about how strides work. 
    I thought that the way that they work was by how much it was stepping in the layer, but having a vector there confuses me. 
    '''
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)


def max_pool(x, stride=[1, 2, 2, 1], pad='SAME', k=[1, 2, 2, 1]):
    ''' stride of pooling  = [1,3,3,1]
    A max pooling layer takes the largest value of the values in the covolutional matrix say [[1 2][8 9]] it takes 9
    This has a stride of 2 and a filter of 2 meaning it jumps 2 steps each time in measure and it takes 2 pixels at a time

    :param x:
    :param stride:
    :param pad:
    :param k: ksize
    :return:
    '''
    '''
    Can you explain how is this a  2X2 pooling layer?
    I am confused about how this [1,2,2,1] represents a stride of 2  
     '''

    return tf.nn.max_pool(x, ksize=k, strides=stride, padding=pad)


job_name = sys.argv[1]
task_number = int(sys.argv[2])
training_epochs = 50
num_worker = 1

# cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"],
#                                 "worker": ["localhost:2223", "localhost:2224"]})

'''Setting up the cluster for workers and the one parameter server. '''
cluster = tf.train.ClusterSpec({"ps": ["mist27-umh.cs.umn.edu:2227"],
                                "worker": ["mist27-umh.cs.umn.edu:2229"]})

server = tf.train.Server(cluster, job_name=job_name, task_index=task_number)
print("Starting server /job:{}/task:{}".format(job_name, task_number))

print("Starting server /job:{}/task:{}".format(job_name, task_number))

# Data loading
(x_image, Y), (X_test, Y_test) = cifar10.load_data()
y_test_vector = to_categorical(Y_test, 10)
y_vector = to_categorical(Y, 10)
y_features = to_categorical(np.arange(10), 10)
print("Image data: cifar10.load_data (50000)")

''' 

I am comfortable enough to be able to recreate AlexNet following the paper, however, I am still uncertain 
about how to go about it with Cifar Dataset

'''
if job_name == "ps":
    server.join()


elif job_name == "worker":

    with tf.device("/job:ps/task:0"):
        tf.set_random_seed(1)
        W_conv1 = weight_variable(
            [11, 11, 3, 96])  # I am not sure about 96 ... As I understand the 96 is the number of Kernels
        b_conv1 = bias_variable([96])
        # Can you explain this bias variable? I think it is 64....


        W_conv2 = weight_variable([5, 5, 96, 256])
        b_conv2 = bias_variable([256])

        W_conv3 = weight_variable([3, 3, 256, 384])
        b_conv3 = bias_variable([384])

        W_conv4 = weight_variable([3, 3, 384, 256])
        b_conv4 = bias_variable([256])

        W_conv5 = weight_variable([3, 3, 256, 256])
        b_conv5 = bias_variable([256])

        W_fc1 = weight_variable([4 * 256, 4096])  ## not sure about that 12*12*256
        b_fc1 = bias_variable([4096])

        # W_fc2 = weight_variable([4096, 4096])
        # b_fc2 = bias_variable([4096])  ## Is this right?

        W_fc3 = weight_variable([4096, 10])
        b_fc3 = bias_variable([10])

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_number, cluster=cluster)):

        global_step = tf.contrib.framework.get_or_create_global_step()

        # Model
        picture = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        ## Here I changed things to the bigger model.

        # train_x = np.zeros((1, 227, 227, 3)).astype(float32)
        # train_y = np.zeros((1, 1000))
        # xdim = train_x.shape[1:]
        # ydim = train_y.shape[1]


        '''
        Here is where the first layer start and the convolution

        The first and second layer are pooled and normalized.

        The third and fourth layer have nothing done to them.

        The fifth layer is pooled. 

        '''
        print("Shape picture: ", picture.get_shape())
        conv_layer1 = tf.nn.relu(
            conv2d(picture, W_conv1, stride=[1, 4, 4, 1]) + b_conv1)  ## chan you just add? don't you have to do np.add?
        print("Shape conv_layer1: ", conv_layer1.get_shape())
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        normalized_layer1 = tf.nn.local_response_normalization(conv_layer1, depth_radius=radius, alpha=alpha, beta=beta,
                                                               bias=bias)
        print("Shape normalized_layer1: ", normalized_layer1.get_shape())
        layer1 = max_pool(normalized_layer1, k=[1, 2, 2, 1], pad='VALID')
        print("Shape layer1: ", layer1.get_shape())
        # https://github.com/guerzh/tf_weights/blob/master/myalexnet_forward.py

        conv_layer2 = tf.nn.relu(conv2d(layer1, W_conv2) + b_conv2)  ## chan you just add? don't you have to do np.add?
        print("Shape conv_layer2: ", conv_layer2.get_shape())

        normalized_layer2 = tf.nn.local_response_normalization(conv_layer2, depth_radius=radius, alpha=alpha, beta=beta,
                                                               bias=bias)
        print("Shape normalized_layer2: ", normalized_layer2.get_shape())

        layer2 = max_pool(normalized_layer2, k=[1, 2, 2, 1], pad='VALID')
        print("Shape layer2 pooled: ", layer2.get_shape())
        layer3 = tf.nn.relu(conv2d(layer2, W_conv3) + b_conv3)
        print("Shape layer3: ", layer3.get_shape())
        layer4 = tf.nn.relu(conv2d(layer3, W_conv4) + b_conv4)
        print("Shape layer4: ", layer4.get_shape())

        conv_layer5 = tf.nn.relu(conv2d(layer4, W_conv5) + b_conv5)
        print("Shape conv_layer5: ", conv_layer5.get_shape())
        # layer5 = max_pool(conv_layer5, k=[1, 1, 1, 1], pad='VALID')
        # print("Shape layer5: ", layer5.get_shape())

        layer5_flat = tf.reshape(conv_layer5, [-1, int(prod(conv_layer5.get_shape()[1:]))])

        fc1 = tf.nn.relu_layer(layer5_flat, W_fc1, b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        dropout_fc1 = tf.nn.dropout(fc1, keep_prob)
        fc3 = tf.nn.xw_plus_b(fc1, W_fc3, b_fc3)

        '''The magestic sofmax is here. '''
        prob = tf.nn.softmax(fc3)

        # Loss function
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc3))
        Optimizer = tf.train.AdamOptimizer(0.001).minimize(Loss, global_step=global_step)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(task_number == 0)) as sess:

        # Train
        start_time = time.time()
        for i in range(training_epochs):
            print("*** epoch %d ***" % i)
            # if i % 10 == 0:
            #     train_accuracy = sess.run(accuracy, feed_dict={layer1: X_test, y_: y_test_vector, keep_prob: 1.0})
            #     print("step %d, test accuracy %g" % (i, train_accuracy))
            for j in range(50):

                # Ask Rankyung what is going on in here


                if (j % num_worker) == task_number:
                    idxfrom = j * 1000
                    idxto = idxfrom + 1000
                    print("[%d] batch %d - %d" % (task_number, idxfrom, idxto))
                    _, cost, step = sess.run([Optimizer, Loss, global_step],
                                             feed_dict={picture: x_image[idxfrom:idxto],
                                                        y_: y_vector[idxfrom:idxto], keep_prob: 0.5})

        elapsed_time = time.time() - start_time
        print("Execution Time: %d seconds" % elapsed_time)

        # Predict
        # correct_num = 0.0
        # testdata_num = len(Y_test)
        # class_num = 10
        # correct_num_perclass = np.zeros([class_num])
        # for idx in range(testdata_num):
        #
        #     pred_attr = sess.run(layer9,
        #                          feed_dict={layer1: np.expand_dims(X_test[idx], axis=0),
        #                                     y_: np.expand_dims(y_test_vector[idx], axis=0),
        #                                     keep_prob: 1.0})
        #
        #     if np.argmax(pred_attr, 1) == Y_test[idx]:
        #         correct_num_perclass[Y_test[idx]] += 1
        #         correct_num += 1
        #
        # print "Accuracy: %f" % (correct_num / testdata_num)
        # print "Correct number of prediction out of 1000 per class"
        # print correct_num_perclass

        test_accuracy = sess.run(accuracy, feed_dict={picture: X_test, y_: y_test_vector, keep_prob: 1.0})
        print("Test Accuracy %g" % test_accuracy)

        print("done")
