import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
import multiprocessing
import time


class Processor:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def __call__(self, path):
        img_color = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2Lab)
        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        img_bw = np.zeros((self.img_size, self.img_size))
        for i in range(0, self.img_size):
            for j in range(0, self.img_size):
                img_bw[i][j] = img_color[i][j][0]
        return img_bw, img_color


class ImageLoader:
    def __init__(self, img_size=256, img_format='jpeg'):
        self.img_size = img_size
        self.img_format = img_format

    def load_img(self, path):
        img_color = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2Lab)

        img_color = cv2.resize(img_color, (self.img_size, self.img_size))
        img_bw = np.zeros((self.img_size, self.img_size))
        for i in range(0, self.img_size):
            for j in range(0, self.img_size):
                img_bw[i][j] = img_color[i][j][0]
                # img_bw[i][j][1] = 0
                # img_bw[i][j][2] = 0
        return img_bw, img_color

    def load_folder(self, folder_path, mp=False):
        examples = []
        labels = []
        print("Loading files from folder...")
        start = time.time()
        files = glob(folder_path + '/*.' + self.img_format)
        if not mp:
            for filename in tqdm(files):
                img_bw, img_color = self.load_img(filename)
                examples.append(img_bw)
                labels.append(img_color)
            print("Images loaded. Time elapsed: ", time.time() - start)
            return np.array(examples), np.array(labels)
        else:
            proc = Processor()
            pool = multiprocessing.Pool()
            result = pool.map(proc, files)
            print("Images loaded. Time elapsed: ", time.time() - start)
            return np.array([i[0] for i in result]), np.array([i[1] for i in result])

    def create_dataset(self, folder_path, mp=False):
        # dataset will be a sequence of tf.Tensor couples
        # (img_bw, img_color)   where the color image is the ground truth
        return tf.data.Dataset.from_tensor_slices(self.load_folder(folder_path, mp))


def main():
    tf.enable_eager_execution()

    loader = ImageLoader(256)
    # np_arr = loader.load_folder(os.pardir + '/test_imgs/a_subfolder')
    # for (imgc, imgbw) in np_arr:
    #     cv2.imshow('img', cv2.cvtColor(cv2.cvtColor(imgc, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2GRAY))
    #     cv2.waitKey()
    #     cv2.imshow('img', cv2.cvtColor(cv2.cvtColor(imgbw, cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2GRAY))
    #     cv2.waitKey()
    dataset = loader.create_dataset(os.pardir + '/test_imgs/a_subfolder')
    for data, g_truth in dataset:
        print(data, g_truth)


if __name__ == '__main__':
    main()

