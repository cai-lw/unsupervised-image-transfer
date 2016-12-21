from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from tools.ops import *
from tools.utils import *

class SVHN(object):
    def __init__(self, sess, image_size = 32, batch_size=64, c_dim=3, y_dim=10,
                 checkpoint_dir = None):
        """
        Args:
            sess: TensorFlow session
            image_size: The size of input images. [32]
            batch_size: The size of batch. Should be specified before training. [64]
            y_dim: Dimension of dim for y. [10]
            c_dim: Dimension of image color. For grayscale input, set to 1. [3]
        """

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def net(self, images, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # 32 * 32 * 3
        h0_conv = relu(conv2d(images, 16, k_h=3, k_w=3, d_h=1, d_w=1, name='h0_conv'))
        h1_conv = relu(conv2d(h0_conv, 16, k_h=3, k_w=3, d_h=1, d_w=1, name='h1_conv'))
        h1_pool = maxpooling2d(h1_conv, k_h=2, k_w=2, step_h=2, step_w=2)

        # 16 * 16 * 16
        h2_conv = relu(conv2d(h1_pool, 32, k_h=3, k_w=3, d_h=1, d_w=1, name='h2_conv'))
        h3_conv = relu(conv2d(h2_conv, 32, k_h=3, k_w=3, d_h=1, d_w=1, name='h3_conv'))
        h3_pool = maxpooling2d(h3_conv, k_h=2, k_w=2, step_h=2, step_w=2)

        # 8 * 8 * 32
        h4_conv = relu(conv2d(h1_pool, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='h4_conv'))
        h5_conv = relu(conv2d(h2_conv, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='h5_conv'))
        h5_pool = maxpooling2d(h3_conv, k_h=2, k_w=2, step_h=2, step_w=2)

        # 4 * 4 * 64
        h6_conv = relu(conv2d(h1_pool, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='h6_conv'))
        h7_conv = relu(conv2d(h2_conv, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='h7_conv'))
        h7_pool = maxpooling2d(h3_conv, k_h=2, k_w=2, step_h=2, step_w=2)

        #  2 * 2 * 128
        h8_conv = relu(conv2d(h1_pool, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='h8_conv'))
        h9_conv = relu(conv2d(h2_conv, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='h9_conv'))
        h9_pool = maxpooling2d(h3_conv, k_h=2, k_w=2, step_h=2, step_w=2)

        # linear ops
        h10_lin = linear(tf.reshape(h9_pool, [self.batch_size, -1]), 256, 'h10_lin')
        res = linear(h10_lin, 10, 'res')

        return res


    def build_model(self):

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, self.c_dim],
                                    name = 'images')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name = 'y')

        self.res = self.net(self.images)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.res, self.y))

        self.train_vars = tf.trainable_variables()

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train SVHN Model"""

        # data_X_extra, data_y_extra = load_image_from_mat(os.path.join(config.src_dir, 'extra_32x32.mat'))
        data_X_train, data_y_train = load_image_from_mat(os.path.join(config.src_dir, 'train_32x32.mat'))
        # data_X_test, data_y_test = load_image_from_mat(os.path.join(config.src_dir, 'test_32x32.mat'))
        # data_X_train = np.concatenate((data_X_train, data_X_extra), axis=0)
        # data_y_train = np.concatenate((data_y_train, data_y_extra), axis=0)
        indices = np.arange(len(data_X_train))
        np.random.shuffle(indices)
        data_X_train = data_X_train[indices]
        data_y_train = data_y_train[indices]

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.loss, var_list = self.train_vars)

        tf.initialize_all_variables().run()
        summary_op = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):

            batch_idxs = len(data_y_train) // config.batch_size

            for idx in xrange(0, batch_idxs):

                batch_images = data_X_train[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_y = data_y_train[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_images = np.array(batch_images) / 127.5 - 1
                batch_images.astype(np.float32)

                # Update network
                self.sess.run(optim, feed_dict = {self.images : batch_images, \
                        self.y : batch_y})

                if np.mod(counter, 30) == 0:
                    summary_str = session.run(summary_op)
                    summary_writer.add_summary(summary_str, total_step)

                    network_error = self.loss.eval({self.images: batch_images, self.y: batch_y})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                        % (epoch, idx, batch_idxs, time.time() - start_time, network_error))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):

        model_name = "SVHN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):

        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
