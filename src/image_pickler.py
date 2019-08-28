import pickle
import os
from typing import List
import cv2
from src.config import imgs_dir
from numpy.core._multiarray_umath import ndarray


def image_pickler(images_folder_path: str, fmt: str):
    names: List[str] = [f for f in os.listdir(images_folder_path) if f.lower().endswith(fmt)]
    images_list: List[ndarray] = []
    for name in names:
        filename: str = os.path.join(images_folder_path, name)
        bgr = cv2.imread(filename)
        images_list.append(bgr)
    pickle_out = open("images.pickle", "wb")
    pickle.dump(images_list, pickle_out)
    pickle_out.close()


def image_unpickler(pickler_file_path):
    pickle_in = open(pickler_file_path, "rb")
    images = pickle.load(pickle_in)
    return images


def main():
    image_pickler(os.pardir + imgs_dir, 'jpeg')
    images = image_unpickler('images.pickle')
    print(images)


if __name__ == '__main__':
    main()