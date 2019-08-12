# from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from glob import glob


class ImageLoader:
    def __init__(self, img_size):
        self.img_size = img_size

    def load_img(self, path):
        img_color = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2Lab)

        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        img_bw = img_color.copy()
        for i in range(0, self.img_size):
            for j in range(0, self.img_size):
                img_bw[i][j] = img_bw[i][j].take(1)
        return img_color, img_bw

    def load_folder(self, folder_path):
        image_list = []
        for filename in glob(folder_path + '/*.jpg'):
            image_list.append(self.load_img(filename))
        return image_list

#
# from PIL import Image
# import glob
# image_list = []
# for filename in glob.glob('yourpath/*.gif'): #assuming gif
#     im=Image.open(filename)
#     image_list.append(im)
def main():

    # Allows INFO logs to be printed
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # Set verbosity of warnings allowed (higher number, less warnings)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    loader = ImageLoader(256)
    # imgc, imgbw = loader.load_img(os.pardir + '/test_imgs/a_subfolder/twitch.jpg')
    # cv2.imshow('img', imgc)
    # cv2.waitKey()
    # cv2.imshow('img', imgbw)
    # cv2.waitKey()
    # list = loader.load_folder(os.pardir + '/test_imgs/a_subfolder')
    # for (imgc, imgbw) in list:
    #     cv2.imshow('img', imgc)
    #     cv2.waitKey()
    #     cv2.imshow('img', imgbw)
    #     cv2.waitKey()




if __name__ == '__main__':
    main()

