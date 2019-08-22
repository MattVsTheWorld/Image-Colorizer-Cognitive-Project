import os
from numpy.core._multiarray_umath import ndarray
from typing import List, Tuple
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import sklearn.neighbors as nn
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
from src.config import data_dir


def load_data(size: int = 64,
              image_folder: str = os.pardir + 'test_imgs/bird',
              fmt: str = '.jpeg') -> ndarray:
    """
    Loads a sample of images
    :param size: width/height to resize images to
    :param image_folder: path to images
    :param fmt: format of images
    :return: X_ab, an ndarray containing the samples in Lab format (a,b channels only)
    """
    names: List[str] = [f for f in os.listdir(image_folder) if f.lower().endswith(fmt)]
    np.random.shuffle(names)
    num_samples: int = 1000
    X_ab: ndarray = np.empty((num_samples, size, size, 2))
    # Take the first num_samples (shuffled) images
    for i in range(num_samples):
        name: str = names[i]
        filename: str = os.path.join(image_folder, name)
        bgr: ndarray = cv2.imread(filename)
        bgr = cv2.resize(bgr, (size, size), cv2.INTER_AREA)
        lab: ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.int32)
        X_ab[i] = lab[:, :, 1:] - 128
    return X_ab


def compute_color_prior(X_ab: ndarray):
    """
    Calculate prior color probability of dataset
    :param X_ab: Sample of images
    """
    q_ab: ndarray(dtype=int, shape=(313, 2)) = np.load(data_dir + "/lab_gamut.npy")
    X_a: ndarray = np.ravel(X_ab[:, :, :, 0])
    X_b: ndarray = np.ravel(X_ab[:, :, :, 1])
    X_ab: ndarray = np.vstack((X_a, X_b)).T

    # Create nearest neighbour instance with index = q_ab
    # Basically each point is its closest representative
    num_neighbours: int = 1
    nearest = nn.NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(q_ab)
    # index and distance of nearest neigh
    dist, idx = nearest.kneighbors(X_ab)

    # Count number of occurrences of each color
    idx: ndarray = np.ravel(idx)
    # Counts how many of each color is present (counting how many times their index is present)
    counts: ndarray = np.bincount(idx)
    # Only retain the indexes of colors that are represented
    distribution: ndarray = np.nonzero(counts)[0]

    prior_prob: ndarray = np.zeros((q_ab.shape[0]))
    # for i in range(q_ab.shape[0]): TODO: check memes
    prior_prob[distribution] = counts[distribution]

    # Transform into probability
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save in data
    np.save(data_dir + "/prior_probability.npy", prior_prob)


def smooth_color_prior(sigma: int = 5):
    """
    Smooth color probability with a gaussian window
    :param sigma: gaussian parameter
    """
    prior_prob: ndarray = np.load(data_dir, "prior_probability.npy")
    # Epsilon piccola a p i a c e r e (avoids 0 values/ NaN)
    prior_prob += 1E-3 * np.min(prior_prob)
    # Renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    # create 1000 evenly spaced numbers [0,312]
    x_coord = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    y_coord = f(x_coord)
    # 2000 points in the window
    window = gaussian(2000, sigma)
    smoothed = convolve(y_coord, window / window.sum(), mode='same')
    fout = interp1d(x_coord, smoothed)
    prior_prob_smoothed: ndarray = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed: ndarray = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    np.save(data_dir + "/prior_prob_smoothed.npy", prior_prob_smoothed)


def compute_prior_factor(gamma: float = 0.5, alpha: int = 1):
    """
    TODO: explain
    :param gamma:
    :param alpha:
    :return:
    """
    prior_prob_smoothed = np.load(data_dir + "prior_prob_smoothed.npy")

    u: ndarray = np.ones_like(prior_prob_smoothed)
    u: ndarray = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    np.save(data_dir + "/prior_factor.npy", prior_factor)


def main():
    image_folder: str = os.pardir + '/test_imgs/bird'
    size: int = 64
    # Load the sample of images
    X_ab = load_data(size, image_folder)
    # Calculate prior probability of color
    compute_color_prior(X_ab)
    smooth_color_prior()
    # TODO: read paper
    compute_prior_factor()


if __name__ == '__main__':
    main()
