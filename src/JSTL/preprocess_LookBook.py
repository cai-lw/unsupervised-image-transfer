import numpy as np
import cv2


if __name__ == '__main__':
    img_prefix = '../../Dataset/lookbook/data/'
    # Load the image
    tmp_count = 0
    for line in open('LookBookList.txt'):
        line = line.split(' ')
        img = cv2.imread(img_prefix + line[0])
        img = cv2.resize(img, (56, 144))
        img = img[np.newaxis, ...]
        if tmp_count == 0:
            img_mat = img
        else:
            img_mat = np.concatenate((img_mat, img), axis = 0)
            print img_mat.shape
        tmp_count += 1
        np.save('../../Dataset/lookbook/data_144_56.npy', img_mat)
