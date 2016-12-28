import numpy as np
import cv2


if __name__ == '__main__':
    img_prefix = '../../Dataset/lookbook/data/'
    # Load the image
    tmp_count = 0
    img_mat = np.zeros((77546, 144, 56, 3), dtype = np.uint8)
    for line in open('LookBookList.txt'):
        line = line.split(' ')
        img = cv2.imread(img_prefix + line[0])
        img = cv2.resize(img, (56, 144))
        img = img[np.newaxis, ...]
        img_mat[tmp_count, ...] = img
        tmp_count += 1
        print tmp_count
    np.save('../../Dataset/lookbook/data_144_56.npy', img_mat)
