import numpy as np
import random
import tensorflow as tf
from JSTL import JSTL

batch_size = 32
CONTRASTIVE_MARGIN = 1.0

def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = np.reshape(source.images[i], (28, 28, 1))
            label = source.labels[i]
            yield image, label

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        label_batch = []
        for _ in range(batch_size):
            image, label = next(data_gen)
            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)


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
    train_left_data = tf.placeholder(tf.float32, shape=(batch_size, 144, 56, 3), name="left")
    train_right_data = tf.placeholder(tf.float32, shape=(batch_size, 144, 56, 3), name="right")
    labels_data = tf.placeholder(tf.int32, shape= batch_size * 2)
    net_left = JSTL({'data': train_left_data})
    net_right = JSTL({'data': train_right_data})
    feature_left = new_left.layers['fc7']
    feature_right = new_right.layers['fc7']

    loss, between_class, within_class = compute_contrastive_loss(feature_left, feature_right, labels_data, CONTRASTIVE_MARGIN)
    opt = tf.train.RMSPropOptimizer(0.001)
    train_op = opt.minimize(loss)

    with tf.Session() as sess:
        # Load the data
        sess.run(tf.initialize_all_variables())
        net.load('JSTL.npy', sess)

        data_gen = gen_data_batch(mnist.train)
        for i in range(1000):
            np_images, np_labels = next(data_gen)
            feed = {images: np_images, labels: np_labels}

            np_loss, np_pred, _ = sess.run([loss, pred, train_op], feed_dict=feed)
            if i % 10 == 0:
                print('Iteration: ', i, np_loss)
