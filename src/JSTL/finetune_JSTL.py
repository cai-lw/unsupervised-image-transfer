import numpy as np
import random
import tensorflow as tf
from JSTL import JSTL

BATCH_SIZE = 32
CONTRASTIVE_MARGIN = 1.0

def read_data(data_path, label_path):
    labels = []
    for line in open(label_path):
        line = line.split(' ')
        img_list.append(line[1])
    data = np.load(data_path)
    return data, labels

def get_batch_pair(data, labels, batch_size, is_target_set_train=True):
    def get_genuine_or_not(data, labels, genuine=True):
        total_labels = 8725
        if genuine:
            index = numpy.random.randint(total_labels)
            # Getting the indexes of the data from a particular client
            indexes = numpy.where(labels == index)[0]
            numpy.random.shuffle(indexes)

            # Picking a pair
            one_left = input_data[indexes[0], :, :, :]
            one_right = input_data[indexes[1], :, :, :]
        else:
            # Picking a pair from different clients
            index = numpy.random.choice(total_labels, 2, replace=False)
            # Getting the indexes of the two clients
            index_left = numpy.where(labels == index[0])[0]
            index_rigth = numpy.where(labels == index[1])[0]
            numpy.random.shuffle(index_left)
            numpy.random.shuffle(index_right)

            # Picking a pair
            data_left = data[index_left[0], :, :, :]
            data_right = data[index_right[0], :, :, :]

        return data_left, data_right

    if is_target_set_train:
            target_data = self.train_data
            target_labels = self.train_labels
        else:
            target_data = self.validation_data
            target_labels = self.validation_labels

        batch_left = numpy.zeros(shape = (batch_size, 144, 56, 3), dtype='float32')
        batch_right = numpy.zeros(shape = (batch_size, 144, 56, 3), dtype='float32')
        batch_labels = numpy.zeros(shape = batch_size, dtype='float32')

        genuine = True
        for i in range(total_data):
            data_left[i, :, :, :], data_right[i, :, :, :] = get_genuine_or_not(target_data, target_labels, genuine=genuine)
            labels_siamese[i] = not genuine
            genuine = not genuine

        return data_left, data_right, labels


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope('euclidean_distance') as scope:
        #d = tf.square(tf.sub(x, y))
        #d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), 1))
        return d


def compute_contrastive_loss(left_feature, right_feature, label, margin, is_target_set_train=True):

    """
    Compute the contrastive loss as in
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2
    OR MAYBE THAT
    L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2
    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin
    **Returns**
     Return the loss operation
    """

    with tf.name_scope("contrastive_loss"):
        label = tf.to_float(label)
        one = tf.constant(1.0)

        d = compute_euclidean_distance(left_feature, right_feature)
        #first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
        #first_part = tf.mul(label, tf.square(d))  # (Y-1)*(d^2)
        between_class = tf.exp(tf.mul(one-label, tf.square(d)))  # (1-Y)*(d^2)
        max_part = tf.square(tf.maximum(margin-d, 0))

        within_class = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)
        #second_part = tf.mul(one-label, max_part)  # (Y) * max((margin - d)^2, 0)

        loss = 0.5 * tf.reduce_mean(within_class + between_class)

        return loss, tf.reduce_mean(within_class), tf.reduce_mean(between_class)


if __name__ == 'main':
    # Siamease place holders - Training
    train_left_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 144, 56, 3), name="left")
    train_right_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 144, 56, 3), name="right")
    labels_data = tf.placeholder(tf.int32, shape= BATCH_SIZE * 2)
    net_left = JSTL({'data': train_left_data})
    net_right = JSTL({'data': train_right_data})
    feature_left = new_left.layers['fc7']
    feature_right = new_right.layers['fc7']

    loss, between_class, within_class = compute_contrastive_loss(feature_left, feature_right, labels_data, CONTRASTIVE_MARGIN)
    opt = tf.train.RMSPropOptimizer(0.001)
    train_op = opt.minimize(loss)

    data, labels = read_data('../../Dataset/lookbook/data_144_56.npy', 'LookBookList.txt')

    with tf.Session() as sess:
        # Load the data
        sess.run(tf.initialize_all_variables())
        net.load('JSTL.npy', sess)

        for i in range(1000):

            batch_left, batch_right, labels = data_shuffler.get_pair(data, labels, BATCH_SIZE)

            np_loss, _ = sess.run([loss, train_op], feed_dict = {train_left_data: batch_left,
                                                                train_right_data: batch_right,
                                                                labels_data: labels)
            if i % 10 == 0:
                print('Iteration: ', i, np_loss)
