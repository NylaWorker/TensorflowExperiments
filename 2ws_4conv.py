# python mp_2wk_2fc.py ps 0
# python mp_2wk_2fc.py worker 0
# python mp_2wk_2fc.py worker 1

"""Rankyung code"""

import sys
import tensorflow as tf
import numpy as np
from tflearn.data_utils import to_categorical
from tflearn.datasets import cifar10
import time

# Configuration
job_name = sys.argv[1]
task_number = int(sys.argv[2])
training_epochs = 50
num_worker = 2

# cluster = tf.train.ClusterSpec({"worker": ["localhost:2223", "localhost:2224"]})
cluster = tf.train.ClusterSpec({"worker": ["mist38-umh.cs.umn.edu:2222", "mist27-umh.cs.umn.edu:2222"]})
server = tf.train.Server(cluster, job_name=job_name, task_index=task_number)

print("Starting server /job:{}/task:{}".format(job_name, task_number))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Data loading
(x_image, Y), (X_test, Y_test) = cifar10.load_data()
y_test_vector = to_categorical(Y_test, 10)
y_vector = to_categorical(Y, 10)
y_features = to_categorical(np.arange(10), 10)
print "Image data: cifar10.load_data (50000)"


if task_number != 0:
    server.join()

with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:0", cluster=cluster)):

    tf.set_random_seed(1)
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Model
    picture = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    tf.set_random_seed(1)

    W_conv1 = weight_variable([11, 11, 3, 96])
    b_conv1 = bias_variable([96])
    print("Shape picture: ", picture.get_shape())
    conv_layer1 = tf.nn.relu(conv2d(picture, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)
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

    W_conv2 = weight_variable([5, 5, 96, 256])
    b_conv2 = bias_variable([256])
    conv_layer2 = tf.nn.relu(conv2d(layer1, W_conv2) + b_conv2)  ## chan you just add? don't you have to do np.add?
    print("Shape conv_layer2: ", conv_layer2.get_shape())
    normalized_layer2 = tf.nn.local_response_normalization(conv_layer2, depth_radius=radius, alpha=alpha, beta=beta,
                                                           bias=bias)
    print("Shape normalized_layer2: ", normalized_layer2.get_shape())
    layer2 = max_pool(normalized_layer2, k=[1, 3, 3, 1], pad='VALID')
    print("Shape layer2 pooled: ", layer2.get_shape())


with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:1", cluster=cluster)):
    W_conv3 = weight_variable([3, 3, 256, 384])
    b_conv3 = bias_variable([384])
    layer3 = tf.nn.relu(conv2d(layer2, W_conv3) + b_conv3)
    print("Shape layer3: ", layer3.get_shape())

    W_conv4 = weight_variable([3, 3, 384, 256])
    b_conv4 = bias_variable([256])
    layer4 = tf.nn.relu(conv2d(layer3, W_conv4) + b_conv4)
    print("Shape layer4: ", layer4.get_shape())
    layer4_flat = tf.reshape(layer4, [-1, int(prod(layer4.get_shape()[1:]))])

    W_fc1 = weight_variable([2304, 4096])  ## not sure about that 12*12*256
    b_fc1 = bias_variable([4096])
    fc1 = tf.nn.relu_layer(layer4_flat, W_fc1, b_fc1)

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
        print "*** epoch %d ***" % i
        # if i % 10 == 0:
        #     train_accuracy = sess.run(accuracy, feed_dict={layer1: X_test, y_: y_test_vector, keep_prob: 1.0})
        #     print("step %d, test accuracy %g" % (i, train_accuracy))
        for j in range(50):
            if (j % num_worker) == task_number:
                idxfrom = j * 1000
                idxto = idxfrom + 1000
                print "[%d] batch %d - %d" % (task_number, idxfrom, idxto)
                _, cost, step = sess.run([Optimizer, Loss, global_step],
                                         feed_dict={layer1: x_image[idxfrom:idxto],
                                                    y_: y_vector[idxfrom:idxto],
                                                    keep_prob: 0.5})


    elapsed_time = time.time() - start_time
    print "Execution Time: %d seconds" % elapsed_time

    test_accuracy = sess.run(accuracy, feed_dict={layer1: X_test, y_: y_test_vector, keep_prob: 1.0})
    print("Test Accuracy %g" % test_accuracy)

    print("done")
