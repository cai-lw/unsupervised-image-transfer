#!/usr/bin/env python
'''Validate a converted Re ID model on LookBook Dataset'''

import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
import cv2
from JSTL import JSTL


def forward(net, model_path, img_list):
    '''Compute the features for the given network and images.'''
    # Get the input node for feeding in the images
    input_node = net.inputs['data']
    # Get the output of the network (class probabilities)
    net_out = net.get_output()
    img_prefix = '../../Dataset/lookbook/data/'
    fout = open('LookBookFeature.txt', 'w')
    with tf.Session() as sess:
        # Load the converted parameters
        net.load(data_path=model_path, session=sess)
        for img_dir in img_list:
            img = cv2.imread(img_prefix + img_dir)
            img = cv2.resize(img, (56, 144))
            img = img.astype('float32')
            img = img - 102
            img = img[np.newaxis, ...]
            # Start the forwading
            feature = sess.run(net_out,feed_dict = {input_node: img})
            fout.write('%s\n'% img_dir)
            for i in range(feature.shape[1]):
                fout.write('%f '% feature[0, i])
            fout.write('\n')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the converted model parameters (.npy)')
    parser.add_argument('img_path', help='images that one wants to calculate features')
    args = parser.parse_args()

    # Define the network
    images = tf.placeholder(tf.float32, [1, 144, 56, 3])
    net = JSTL({'data' : images})

    # Load the image
    img_list = []
    for line in open(args.img_path):
        line = line.split(' ')
        img_list.append(line[0])

    # Evaluate its performance on the LookBook Dataset
    forward(net, args.model_path, img_list)


if __name__ == '__main__':
    main()
