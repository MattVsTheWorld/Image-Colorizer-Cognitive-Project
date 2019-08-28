import os
import cv2
import numpy as np
import sklearn.neighbors as nn
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.utils import Sequence
sys.stderr = stderr
from trainer.config import percentage_training, batch_size, img_rows, img_cols, imgs_dir, train_set_dim, nb_neighbors
from math import floor

def get_soft_encoding(image_ab, nn_finder, num_q):
    # take shape of first two
    height, width = image_ab.shape[:2]

    # flatten ndarray of the two channels
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    # Metti a fianco
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors (of the color gamut)
    dist_neigh, idx_neigh = nn_finder.kneighbors(ab)
    # Smooooooth weights with gaussian kernel
    sigma = 5
    weights = np.exp(-dist_neigh ** 2 / (2 * sigma ** 2))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    # ----
    # Reshape y
    # shape[0] is length of one of the channels (a)
    y = np.zeros((ab.shape[0], num_q))
    # create indexes from 0 to ab.shape[0]
    # put them in an array with a new axis added
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    # give each color a weight corresponding to how far are its closest 5 neighbours
    y[idx_pts, idx_neigh] = weights
    # num_q (313) as third shape; only num_neighbours (5) will have a non-0 value
    # y = np.reshape(y, (height, width, num_q))
    y = y.reshape(height, width, num_q)

    return y


class DataGenSequence(Sequence):
    def __init__(self, usage, images):
        # Train or validation
        self.usage = usage
        self.images = images

        if usage == 'train':
            self.num_img = percentage_training
        else:
            self.num_img = 1 - percentage_training
            self.images = list(reversed(self.images))

        self.upper_bound = floor(len(self.images) * self.num_img)
        self.images = self.images[:self.upper_bound]

        np.random.shuffle(self.images)
        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.num_q = q_ab.shape[0]

        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        # Number of batches
        return int(np.ceil(len(self.images) / float(batch_size)))

    def __getitem__(self, idx):

        # First element of the batch
        i = idx * batch_size

        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4
        # Batch is either full or partial (last batch)
        length = min(batch_size, (len(self.images) - i))

        # e.g. shape= (32, 256, 256, 1)
        batch_x = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        # e.g. shape= (32, 64, 64, 313)
        batch_y = np.empty((length, out_img_rows, out_img_cols, self.num_q), dtype=np.float32)
        # TODO: remove
        # np.set_printoptions(threshold=sys.maxsize)
        for i_batch in range(length):
            bgr = cv2.resize(self.images[i], (img_rows, img_cols), cv2.INTER_CUBIC)
            # cv2.imshow('lol', bgr)
            # cv2.waitKey()

            gray = cv2.resize(cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY), (img_rows, img_cols), cv2.INTER_CUBIC)

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            x = gray / 255.

            out_lab = cv2.resize(lab, (out_img_rows, out_img_cols), cv2.INTER_CUBIC)
            # rows, columns, L a b; skip L
            # Before: 42 <=a<= 226, 20 <=b<= 223
            # After: -86 <=a<= 98, -108 <=b<= 95
            out_ab = out_lab[:, :, 1:].astype(np.int32) - 128
            # TODO: remove
            # print(out_ab)
            y = get_soft_encoding(out_ab, self.nn_finder, self.num_q)

            if np.random.random_sample() > 0.5:
                # x is gray normalized
                x = np.fliplr(x)
                y = np.fliplr(y)

            # populate batches
            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        # print(batch_y.shape)
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.images)


def train_gen(images):
    return DataGenSequence('train', images)


def valid_gen(images):
    return DataGenSequence('valid', images)

'''
def split_data(image_folder: str, fmt: str):
    names: List[str] = [f for f in os.listdir(image_folder) if f.lower().endswith(fmt)]
    # Number of samples
    num_samples: int = len(names)
    print('num_samples: ' + str(num_samples))

    # Number of train/validation images
    num_train_samples: int = int(num_samples * percentage_training)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples: int = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))

    # Pick random validation file names
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    with open('image_names/valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('image_names/train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))

    with open('image_names/valid_num.txt', 'w') as file:
        file.write(str(num_valid_samples))

    with open('image_names/train_num.txt', 'w') as file:
        file.write(str(num_train_samples))
'''
'''
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
    # TODO: moved

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
    'split_data(image_folder, fmt)'

'''
if __name__ == '__main__':
    main()
