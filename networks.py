import tensorflow as tf
from helpers import IM_SIZE, make_tst_train, create_batches, fetch_batch
from tqdm import tqdm

num_epochs = 5
import numpy as np


def basic_net(img):
    my_net = tf.layers.conv2d(img, filters=32, kernel_size=3, activation=tf.nn.relu, strides=(2, 2),
                              name='conv1')
    my_net = tf.layers.conv2d(my_net, filters=64, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv2')
    my_net = tf.reshape(my_net, [-1, 24 * 24 * 64])
    my_net = tf.layers.dense(my_net, 1000, activation=tf.nn.relu, name='dense1')
    my_net = tf.layers.dense(my_net, 2, activation=None, name='dense2')
    return my_net


def basic_net_with_bn(img):
    my_net = tf.layers.conv2d(img, filters=32, kernel_size=3, activation=tf.nn.relu, strides=(2, 2),
                              name='conv1')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=64, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv2')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.reshape(my_net, [-1, 24 * 24 * 64])
    my_net = tf.layers.dense(my_net, 1000, activation=tf.nn.relu, name='dense1')
    my_net = tf.layers.dense(my_net, 2, activation=None, name='dense2')
    return my_net


def basic_conv_net(img):
    my_net = tf.layers.conv2d(img, filters=32, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv1')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=64, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv2')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=128, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv3')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=256, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv4')
    my_net = tf.layers.batch_normalization(my_net)  # 5x5
    my_net = tf.layers.average_pooling2d(my_net, pool_size=[5, 5], strides=1)
    my_net = tf.layers.conv2d(my_net, filters=2, kernel_size=1, activation=None)
    my_net = tf.reshape(my_net, [-1, 2])
    return my_net


def basic_res_net(img):
    my_net = tf.layers.conv2d(img, filters=32, kernel_size=3, activation=tf.nn.relu, strides=(2, 2),
                              name='conv1', padding='same') + \
             tf.layers.conv2d(img, filters=32, kernel_size=1, strides=(2, 2), activation=None, name='res1')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=64, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv2',
                              padding='same') + \
             tf.layers.conv2d(my_net, filters=64, kernel_size=1, strides=(2, 2), activation=None, name='res2')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=128, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv3',
                              padding='same') + \
             tf.layers.conv2d(my_net, filters=128, kernel_size=1, strides=(2, 2), activation=None, name='res3')
    my_net = tf.layers.batch_normalization(my_net)
    my_net = tf.layers.conv2d(my_net, filters=256, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv4',
                              padding='same') + \
             tf.layers.conv2d(my_net, filters=256, kernel_size=1, strides=(2, 2), activation=None, name='res4')
    my_net = tf.layers.batch_normalization(my_net)  # 5x5
    my_net = tf.layers.average_pooling2d(my_net, pool_size=[7, 7], strides=1)
    my_net = tf.layers.conv2d(my_net, filters=2, kernel_size=1, activation=None)
    my_net = tf.reshape(my_net, [-1, 2])
    return my_net

def basic_dense_net(img):
    my_net = tf.layers.conv2d(img, filters=32, kernel_size=3, activation=tf.nn.relu, strides=(2, 2), name='conv1')
    my_net_1 = tf.layers.batch_normalization(my_net)

    my_net = tf.reshape(my_net, [-1, 2])
    return my_net


def easy_network():
    sess = tf.InteractiveSession()

    x_in = tf.placeholder(tf.float32, shape=[None, IM_SIZE, IM_SIZE])
    x_img = tf.reshape(x_in, shape=[-1, IM_SIZE, IM_SIZE, 1])
    y_in = tf.placeholder(tf.float32, shape=[None, 2])
    y_out = basic_res_net(x_img)

    err = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_in)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step = optimizer.minimize(err)
    correct_prediction = tf.equal(tf.argmax(y_in, 1), tf.argmax(y_out, 1))
    sess.run(tf.global_variables_initializer())

    train_batches, test_batches = create_batches()
    for e in tqdm(range(num_epochs)):
        for train_batch in train_batches:
            im_batch, lab_batch = fetch_batch(train_batch)
            _, correct = sess.run([train_step, correct_prediction], feed_dict={x_in: im_batch, y_in: lab_batch})

            # print(np.mean(correct))

    test_acc = 0
    for test_batch in test_batches:
        im_batch, lab_batch = fetch_batch(test_batch)
        correct = sess.run([correct_prediction], feed_dict={x_in: im_batch, y_in: lab_batch})
        test_acc += np.mean(correct) / len(test_batches)
    print('final test accuracty is: ', test_acc)
    sess.close()
    return test_acc


if __name__ == '__main__':
    easy_network()
