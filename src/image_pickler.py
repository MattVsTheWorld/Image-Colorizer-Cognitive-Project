import pickle
import os
from typing import List
import cv2
from numpy.core._multiarray_umath import ndarray
from google.cloud import storage
from tensorflow.python.lib.io import file_io
import cloudstorage as gcs


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


def gcs_image_unpickler(pickler_file_path):
    #client = storage.Client.from_service_account_json(os.pardir + '\CS-Project-18e33cb1d7f4.json')
    #bucket = client.get_bucket('images_data')
    #blob = bucket.get_blob('images.pickle')
    images = pickle.load(infile)
    return images


def main():
    # image_pickler(os.pardir + '/test_imgs/flower', 'jpg')
    # images = image_unpickler('images.pickle')
    images = gcs_image_unpickler('ahah')
    print(images)



if __name__ == '__main__':
    main()