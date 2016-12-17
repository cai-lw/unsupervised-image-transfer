from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class CrossDomainGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

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

        self.src_images = tf.placeholder(tf.float32, [self.batch_size] + [self.src_size, self.src_size, self.src_c_dim],
                                    name = 'src_images')
        self.tgt_images = tf.placeholder(tf.float32, [self.batch_size] + [self.tgt_size, self.tgt_size, self.tgt_c_dim],
                                    name = 'tgt_images')

        self.src_images_F = self.f_function(self.src_images)
        self.src_images_FG = self.generator(self.src_images_F)
        self.src_images_FGF = self.f_function(self.src_images_FG, reuse=True)

        self.tgt_images_F = self.f_function(self.tgt_images, reuse=True)
        self.tgt_images_FG = self.generator(self.tgt_images_F, reuse=True)

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

        self.GANG_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_1,
                np.array([np.array([0.0, 0.0, 1.0]) for i in range(self.batch_size)]))) \
                + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.D_logits_2,
                np.array([np.array([0.0, 0.0, 1.0]) for i in range(self.batch_size)])))
        self.CONST_loss =  self.CONST_weight * tf.reduce_mean(tf.squared_difference(self.src_images_F, self.src_images_FGF))
        self.TID_loss = self.TID_weight * tf.reduce_mean(tf.squared_difference(self.tgt_images, self.tgt_images_FG))
        # total variation denoising
        shape = tf.shape(self.image)
        self.TV_loss = self.TV_weight * (tf.reduce_mean(tf.squared_difference(self.tgt_images_FG[:,1:,:,:], self.tgt_images_FG[:,:shape[1]-1,:,:])) +
                tf.reduce_mean(tf.squared_difference((self.tgt_images_FG[:,:,1:,:], self.tgt_images_FG[:,:,:shape[2]-1,:]))) +
                tf.reduce_mean(tf.squared_difference(self.src_images_FG[:,1:,:,:], self.src_images_FG[:,:shape[1]-1,:,:])) +
                        tf.reduce_mean(tf.squared_difference((self.src_images_FG[:,:,1:,:], self.src_images_FG[:,:,:shape[2]-1,:]))))
        self.G_loss = self.GANG_loss + self.CONST_loss + self.TID_loss + self.TV_loss

        t_vars = tf.trainable_variables()

        self.D_vars = [var for var in t_vars if 'd_' in var.name]
        self.G_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train CrossDomainGAN"""

        src_data_X, src_data_y = load_image_from_mat(os.path.join(config.src_dir, 'extra_32x32.mat'))
        tgt_data_X, tgt_data_y = load_mnist(config.tgt_dir)

        D_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.D_loss, var_list = self.d_vars)
        G_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                          .minimize(self.G_loss, var_list = self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):

            batch_idxs = min(len(self.src_images), len(self.tgt_images)) // config.batch_size

            for idx in xrange(0, batch_idxs):

                batch_src_images = src_data_X[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_tgt_images = tgt_data_X[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_src_images = np.array(batch_src_images) / 127.5 - 1
                batch_tgt_images = np.array(batch_tgt_images) / 127.5 - 1
                batch_src_images.astype(np.float32)
                batch_tgt_images.astype(np.float32)

                # Update G network
                _, summary_str = self.sess.run([G_optim, self.G_sum], feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})
                self.writer.add_summary(summary_str, counter)

                # Update D network
                _, summary_str = self.sess.run([D_optim, self.D_sum], feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([G_optim, self.G_sum], feed_dict = {self.src_images : batch_src_images, \
                        self.tgt_images : batch_tgt_images})
                self.writer.add_summary(summary_str, counter)

                D_error = self.D_loss.eval({self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})
                G_error = self.G_loss.eval({self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})
                GAND_error = self.GANG_loss.eval({self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})
                CONST_error = self.CONST_loss.eval({self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})
                TID_error = self.TID_loss.eval({self.src_images: batch_src_images, self.tgt_images: batch_tgt_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:sample_labels}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                        )
                    save_images(samples, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        if not self.y_dim:
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def f_function(self, image, reuse = False):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(image, self.df_dim, name='f_h0_conv'))
        h1 = lrelu(self.f_bn1(conv2d(h0, self.df_dim*2, name='f_h1_conv')))
        h2 = lrelu(self.f_bn2(conv2d(h1, self.df_dim*4, name='f_h2_conv')))
        h3 = lrelu(self.f_bn3(conv2d(h2, self.df_dim*8, name='f_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return h4

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:

            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                            [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
