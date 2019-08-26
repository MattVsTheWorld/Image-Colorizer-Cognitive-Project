from numpy.core._multiarray_umath import ndarray
from typing import List, Tuple
import os
import random
from random import shuffle
import cv2
import numpy as np
import sklearn.neighbors as nn
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.utils import Sequence
sys.stderr = stderr
from src.config import percentage_training, batch_size, img_rows, img_cols, imgs_dir, train_set_dim
from glob import glob
import shutil
from tqdm import tqdm


def get_soft_encoding(image_ab, nn_finder, num_q) -> ndarray:
    height: int
    width: int
    # take shape of first two
    height, width = image_ab.shape[:2]

    # flatten ndarray of the two channels
    a: ndarray = np.ravel(image_ab[:, :, 0])
    b: ndarray = np.ravel(image_ab[:, :, 1])
    # Metti a fianco
    ab: ndarray = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors (of the color gamut)
    dist_neigh, idx_neigh = nn_finder.kneighbors(ab)
    # Smooooooth weights with gaussian kernel
    sigma: int = 5
    weights = np.exp(-dist_neigh ** 2 / (2 * sigma ** 2))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    # ----
    # Reshape y
    # shape[0] is length of one of the channels (a)
    y: ndarray = np.zeros((ab.shape[0], num_q))
    # create indexes from 0 to ab.shape[0]
    # put them in an array with a new axis added
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    # give each color a weight corresponding to how far are its closest 5 neighbours
    y[idx_pts, idx_neigh] = weights
    # num_q (313) as third shape; only num_neighbours (5) will have a non-0 value
    y = np.reshape(y, (height, width, num_q))
    return y


class DataGenSequence(Sequence):
    def __init__(self, usage, image_folder):
        # Train or validation
        self.usage: str = usage
        self.image_folder = image_folder

        if usage == 'train':
            names_file: str = 'image_names/train_names.txt'
        else:
            names_file: str = 'image_names/valid_names.txt'

        with open(names_file, 'r') as f:
            self.names: List[str] = f.read().splitlines()

        # np.random.shuffle(self.names)

        # Load the array of quantized ab value
        q_ab: ndarray(dtype=int, shape=(313, 2)) = np.load("data/lab_gamut.npy")
        self.num_q: int = q_ab.shape[0]

        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(algorithm='ball_tree').fit(q_ab)

    def __len__(self) -> int:
        # Number of batches
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        """
        Return an ndarray containing the images contained in batch n°idx
        :param idx: index of the batch
        :return: ndarray containing the batch
        """
        # First element of the batch
        i: int = idx * batch_size

        out_img_rows: int
        out_img_cols: int
        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4
        # Batch is either full or partial (last batch)
        length: int = min(batch_size, (len(self.names) - i))

        # e.g. shape= (32, 256, 256, 1)
        batch_x: ndarray = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        # e.g. shape= (32, 64, 64, 313)
        batch_y: ndarray = np.empty((length, out_img_rows, out_img_cols, self.num_q), dtype=np.float32)

        for i_batch in range(length):
            name: str = self.names[i]
            filename: str = os.path.join(self.image_folder, name)
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            bgr: ndarray = cv2.resize(cv2.imread(filename), (img_rows, img_cols))
            gray: ndarray = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (img_rows, img_cols))
            lab: ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            # Normalize
            x: ndarray = gray / 255.

            out_lab: ndarray = cv2.resize(lab, (out_img_rows, out_img_cols), cv2.INTER_AREA)
            # rows, columns, L a b; skip L
            out_ab: ndarray = out_lab[:, :, 1:].astype(np.int32) - 128

            y: ndarray = get_soft_encoding(out_ab, self.nn_finder, self.num_q)

            # if np.random.random_sample() > 0.5:
            #     # x is gray normalized
            #     x = np.fliplr(x)
            #     y = np.fliplr(y)

            # populate batches
            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen(image_folder: str) -> DataGenSequence:
    return DataGenSequence('train', image_folder)


def valid_gen(image_folder: str) -> DataGenSequence:
    return DataGenSequence('valid', image_folder)


def split_data(image_folder: str, fmt: str):
    names: List[str] = [f for f in os.listdir(image_folder) if f.lower().endswith(fmt)]
    # Number of samples
    num_samples: int = len(names)           # 2601
    print('num_samples: ' + str(num_samples))

    # Number of train/validation images
    num_train_samples: int = int(num_samples * percentage_training)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples: int = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))

    # Pick random validation file names
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    # shuffle(valid_names)
    # shuffle(train_names)

    with open('image_names/valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('image_names/train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))

    with open('image_names/valid_num.txt', 'w') as file:
        file.write(str(num_valid_samples))

    with open('image_names/train_num.txt', 'w') as file:
        file.write(str(num_train_samples))


def generate_dataset():
    source_folder: str = os.pardir + '/imagenet/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/train'
    destination_folder: str = os.pardir + imgs_dir
    folder_list: List[str] = next(os.walk(source_folder))[1]

    # Clear folder
    print("Clearing folder...")
    for the_file in tqdm(os.listdir(destination_folder)):
        file_path = os.path.join(destination_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print("\nDone")

    # avg size of img = 200kb
    # print("Fetching images", sep=' ', end='')
    total_size: int = 0     # current byte size of folder
    print("Fetching dataset...")
    pbar = tqdm(total=train_set_dim)
    while total_size < (train_set_dim * 2**20):
        chosen_one: str = random.choice(folder_list)
        img_path = random.choice(glob(source_folder + '/' + chosen_one + '/*.jpeg'))
        size = os.path.getsize(img_path)
        total_size += size
        pbar.update(size / 2**20)
        shutil.copy(img_path, destination_folder)
    pbar.close()
    print("\nDone")


def main():
    generate_dataset()
    image_folder: str = os.pardir + imgs_dir
    fmt: str = '.jpeg'
    split_data(image_folder, fmt)


if __name__ == '__main__':
    main()
