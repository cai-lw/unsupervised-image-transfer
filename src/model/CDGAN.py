from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from model.SVHN import SVHN

from tools.ops import *
from tools.utils import *

class CrossDomainGAN(object):
    def __init__(self, sess, image_size=108,
                 batch_size=64, sample_size = 64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 CONST_weight=1, TID_weight=1, TV_weight=1,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            image_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = image_size

        self.y_dim = y_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.CONST_weight = CONST_weight
        self.TID_weight = TID_weight
        self.TV_weight = TV_weight

        self.dataset_name = dataset_name

        # batch normalization : deals with poor initialization helps gradient flow
        self.D_bn1 = batch_norm(name = 'D_bn1')
        self.D_bn2 = batch_norm(name = 'D_bn2')
        self.D_bn3 = batch_norm(name = 'D_bn3')

        self.G_bn0 = batch_norm(name = 'G_bn0')
        self.G_bn1 = batch_norm(name = 'G_bn1')
        self.G_bn2 = batch_norm(name = 'G_bn2')
        self.G_bn3 = batch_norm(name = 'G_bn3')

        self.F_bn1 = batch_norm(name='F_bn1')
        self.F_bn2 = batch_norm(name='F_bn2')
        self.F_bn3 = batch_norm(name='F_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.src_images = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, self.c_dim],
                                    name = 'src_images')
        self.tgt_images = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, self.c_dim],
                                    name = 'tgt_images')

        self.f_model = SVHN(self.sess, image_size=self.image_size, batch_size=self.batch_size, c_dim=self.c_dim, checkpoint_dir=self.checkpoint_dir)
        if self.f_model.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        self.f_function = self.f_model.features

        self.src_images_F = self.f_function(self.src_images)
        self.src_images_FG = self.generator(self.src_images_F)
        self.src_images_FGF = self.f_function(self.src_images_FG)

        self.tgt_images_F = self.f_function(self.tgt_images)
        self.tgt_images_FG = self.generator(self.tgt_images_F, reuse=True)
        self.gen_images_sum = tf.image_summary('gen_images', self.tgt_images_FG)

        self.D_1, self.D_logits_1 = self.discriminator(self.src_images_FG)
        self.D_2, self.D_logits_2 = self.discriminator(self.tgt_images_FG, reuse=True)
        self.D_3, self.D_logits_3 = self.discriminator(self.tgt_images, reuse=True)

        self.D_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_1,
                np.array([np.array([1.0, 0.0, 0.0]) for i in range(self.batch_size)])))
        self.D_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_2,
                np.array([np.array([0.0, 1.0, 0.0]) for i in range(self.batch_size)])))
        self.D_loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_3,
                np.array([np.array([0.0, 0.0, 1.0]) for i in range(self.batch_size)])))

        self.D_loss = self.D_loss_1 + self.D_loss_2 + self.D_loss_3
        self.D_loss_sum = tf.scalar_summary('D_loss', self.D_loss)

        self.GANG_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_1,
                np.array([np.array([0.0, 0.0, 1.0]) for i in range(self.batch_size)]))) \
                + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_2,
                np.array([np.array([0.0, 0.0, 1.0]) for i in range(self.batch_size)])))
        self.GANG_loss_sum = tf.scalar_summary('GANG_loss', self.GANG_loss)

        self.CONST_loss = tf.reduce_mean(tf.squared_difference(self.src_images_F, self.src_images_FGF))
        self.CONST_loss_sum = tf.scalar_summary('CONST_loss', self.CONST_loss)

        self.TID_loss = tf.reduce_mean(tf.squared_difference(self.tgt_images, self.tgt_images_FG))
        self.TID_loss_sum = tf.scalar_summary('TID_loss', self.TID_loss)

        # total variation denoising
        self.TV_loss = (tf.reduce_mean(tf.squared_difference(self.tgt_images_FG[:,1:,:,:], self.tgt_images_FG[:,:-1,:,:])) +
                tf.reduce_mean(tf.squared_difference(self.tgt_images_FG[:,:,1:,:], self.tgt_images_FG[:,:,:-1,:])) +
                tf.reduce_mean(tf.squared_difference(self.src_images_FG[:,1:,:,:], self.src_images_FG[:,:-1,:,:])) +
                        tf.reduce_mean(tf.squared_difference(self.src_images_FG[:,:,1:,:], self.src_images_FG[:,:,:-1,:])))
        self.TV_loss_sum = tf.scalar_summary('TV_loss', self.TV_loss)

        self.G_loss = self.GANG_loss + self.CONST_weight * self.CONST_loss + self.TID_weight * self.TID_loss + self.TV_weight * self.TV_loss
        self.G_loss_sum = tf.scalar_summary('G_loss', self.G_loss)

        self.all_sum = tf.merge_summary([self.D_loss_sum, self.GANG_loss_sum, self.CONST_loss_sum,
            self.TID_loss_sum, self.TV_loss_sum, self.G_loss_sum])

        t_vars = tf.trainable_variables()

        self.D_vars = [var for var in t_vars if 'd_' in var.name]
        self.G_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train CrossDomainGAN"""

        src_data_X_train, src_data_y_train = load_image_from_mat(os.path.join(config.src_dir, 'train_32x32.mat'))
        src_data_X_test, src_data_y_test = load_image_from_mat(os.path.join(config.src_dir, 'test_32x32.mat'))
        tgt_data_X_train, tgt_data_y_train = load_mnist(config.tgt_dir, part="train")
        tgt_data_X_test, tgt_data_y_test = load_mnist(config.tgt_dir, part="test")

        D_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.D_loss, var_list = self.D_vars)
        G_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.G_loss, var_list = self.G_vars)

        writer = tf.train.SummaryWriter(config.log_dir, self.sess.graph)

        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):

            batch_idxs = min(src_data_X_train.shape[0], tgt_data_X_train.shape[0]) // config.batch_size

            for idx in xrange(0, batch_idxs):

                batch_src_images = src_data_X_train[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_tgt_images = tgt_data_X_train[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_src_images = batch_src_images.astype(np.float32) / 127.5 - 1
                batch_tgt_images = batch_tgt_images.astype(np.float32) / 127.5 - 1

                # Update G network
                self.sess.run(G_optim, feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})

                # Update D network
                self.sess.run(D_optim, feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})

                # Update G network
                self.sess.run(G_optim, feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})

                summary = self.sess.run(self.all_sum, feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})
                writer.add_summary(summary, counter)

                D_error, G_error = self.sess.run([self.D_loss, self.G_loss],
                    feed_dict = {self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx+1, batch_idxs,
                        time.time() - start_time, D_error, G_error))

                if np.mod(counter, 100) == 0:
                    test_batch_idxs = min(src_data_X_test.shape[0], tgt_data_X_test.shape[0]) // config.batch_size
                    test_d_loss = 0
                    test_g_loss = 0
                    for idx in xrange(0, test_batch_idxs):
                        batch_src_test = src_data_X_test[idx * config.batch_size : (idx + 1) * config.batch_size]
                        batch_tgt_test = tgt_data_X_test[idx * config.batch_size : (idx + 1) * config.batch_size]
                        d_loss, g_loss = self.sess.run(
                            [self.D_loss, self.G_loss],
                            feed_dict={self.src_images: batch_src_test, self.tgt_images: batch_tgt_test}
                        )
                        test_d_loss += d_loss
                        test_g_loss += g_loss
                        if idx * 5 % test_batch_idxs < 5:
                            samples = self.sess.run(self.src_images_FG,
                                feed_dict={self.src_images: batch_src_test, self.tgt_images: batch_tgt_test})
                            batch_mosaic_size = [int(np.ceil(np.sqrt(config.batch_size)))] * 2
                            save_images(batch_src_test, batch_mosaic_size,
                                './{}/{:04d}_{:04d}_src.png'.format(config.sample_dir, counter, idx))
                            save_images(samples, batch_mosaic_size,
                                './{}/{:04d}_{:04d}_gen.png'.format(config.sample_dir, counter, idx))
                    test_d_loss /= test_batch_idxs
                    test_g_loss /= test_batch_idxs
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (test_d_loss, test_g_loss))

                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.D_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.D_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.D_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 3, 'd_h3_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.D_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat(1, [h1, y])

                h2 = lrelu(self.D_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat(1, [h2, y])

                h3 = linear(h2, 3, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            if not self.y_dim:
                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.G_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.G_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(h1,
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.G_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(h2,
                    [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.G_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3,
                    [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.G_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.G_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.G_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))


    def sampler(self, z, y=None):
        with tf.variable_scope("generator", reuse=True):
            if not self.y_dim:

                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
                h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                                [-1, s16, s16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.G_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.G_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.G_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.G_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.G_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.G_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.G_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))


    def save(self, checkpoint_dir, step):
        model_name = "CrossDomainGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading CDGAN checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
