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
                img_bw[i][j] = img_bw[i][j].take(0)
        return img_color, img_bw

    def load_folder(self, folder_path):
        image_list = []
        for filename in glob(folder_path + '/*.jpg'):
            image_list.append(self.load_img(filename))
        return image_list

def main():
    loader = ImageLoader(256)
    # imgc, imgbw = loader.load_img(os.pardir + '/test_imgs/a_subfolder/twitch.jpg')
    # cv2.imshow('img', imgc)
    # cv2.waitKey()
    # cv2.imshow('img', imgbw)
    # cv2.waitKey()
    list = loader.load_folder(os.pardir + '/test_imgs/a_subfolder')
    for (imgc, imgbw) in list:
        cv2.imshow('img', imgc)
        cv2.waitKey()
        cv2.imshow('img', imgbw)
        cv2.waitKey()


if __name__ == '__main__':
    main()

