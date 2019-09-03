import pickle
import os
import cv2
from trainer.config import imgs_dir
from google.cloud import storage
import numpy as np


def image_pickler(images_folder_path, fmt):
    names = [f for f in os.listdir(images_folder_path) if f.lower().endswith(fmt)]
    images_list = []
    for name in names:
        filename = os.path.join(images_folder_path, name)
        bgr = cv2.imread(filename)
        images_list.append(bgr)
    pickle_out = open("images.pickle", "wb")
    pickle.dump(images_list, pickle_out, protocol=2)
    pickle_out.close()


def image_unpickler(pickler_file_path):
    pickle_in = open(pickler_file_path, "rb")
    images = pickle.load(pickle_in)
    return images


def gcs_image_unpickler(pickler_file_path):
    client = storage.Client()  # .Client.from_service_account_json('CS-Project-18e33cb1d7f4.json')
    bucket = client.get_bucket('images_regional')
    blob = bucket.get_blob('images.pickle')
    string = blob.download_as_string()
    # print(string)
    images = pickle.loads(string)
    return images


def npy_pickler(file_name):
    npy = np.load(os.path.join(file_name + '.npy'))
    pickle_out = open(file_name + '.pickle', "wb")
    pickle.dump(npy, pickle_out, protocol=2)
    pickle_out.close()


def gcs_npy_unpickler(file_name):
    client = storage.Client() #.from_service_account_json('CS-Project-18e33cb1d7f4.json')
    bucket = client.get_bucket('images_regional')
    blob = bucket.get_blob(file_name)
    string = blob.download_as_string()
    npy = pickle.loads(string)
    return npy


def main():
    # image_pickler(os.pardir + '/test_images/flower', 'jpg')
    # images = image_unpickler('images.pickle')
    #images = gcs_image_unpickler('ahah')
    #print(images)
    npy_pickler('pts_in_hull')

if __name__ == '__main__':
    main()
