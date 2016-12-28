import os
import numpy as np
from model.SVHN import SVHN
from tools.utils import pp
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 32, "The size of the images [32]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("y_dim", 10, "Dimension of class. [10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the tensorflow summaries [logs]")
flags.DEFINE_string("src_dir", "../Dataset/SVHN", "Directory name to dataset [../Dataset/SVHN]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    with tf.Session() as sess:
        model = SVHN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, c_dim=FLAGS.c_dim,
                     y_dim=FLAGS.y_dim, checkpoint_dir = FLAGS.checkpoint_dir)
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.load(FLAGS.checkpoint_dir)

if __name__ == '__main__':
    tf.app.run()
