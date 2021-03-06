import os
import cv2
import numpy as np
import sklearn.neighbors as nn
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
from config import imgs_dir, fmt, num_colors
from config import data_dir as abs_data_dir
from tqdm import tqdm


def load_data(size, image_folder=imgs_dir):
    # """
    # Loads a sample of images
    # :param size: width/height to resize images to
    # :param image_folder: path to images
    # :param fmt: format of images
    # :return: images_ab, an ndarray containing the samples in Lab format (a,b channels only)
    # """
    names = [f for f in os.listdir(image_folder) if f.lower().endswith(fmt)]
    np.random.shuffle(names)
    num_samples = len(names)  # prior_sample_size
    print("Creating prior based on " + str(num_samples) + " images")
    images_ab = np.empty((num_samples, size, size, 2))
    # Take the first num_samples (shuffled) images
    for i in tqdm(range(num_samples)):
        name = names[i]
        filename = os.path.join(image_folder, name)
        bgr = cv2.imread(filename)
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.int32)
        images_ab[i] = lab[:, :, 1:] - 128

    return images_ab


def compute_color_prior(images_ab, data_dir=abs_data_dir):
    # """
    # Calculate prior color probability of dataset
    # :param X_ab: Sample of images
    # """
    q_ab = np.load(os.path.join(data_dir, "lab_gamut.npy"))
    images_a = np.ravel(images_ab[:, :, :, 0])
    images_b = np.ravel(images_ab[:, :, :, 1])
    images_ab = np.vstack((images_a, images_b)).T

    # Create nearest neighbour instance with index = q_ab
    # Basically each point is its closest representative
    num_neighbours = 1
    nearest = nn.NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(q_ab)
    # index and distance of nearest neigh
    dist, idx = nearest.kneighbors(images_ab)

    # Count number of occurrences of each color
    idx = np.ravel(idx)

    # Counts how many of each color is present (counting how many times their index is present)
    counts = np.bincount(idx, minlength=num_colors).astype(np.float32)
    # Epsilon small to taste (avoids 0 values/ NaN)
    counts += 1E-6
    # Transform into probability
    prior_prob = counts / (1.0 * np.sum(counts))

    # Save in data
    np.save(data_dir + "prior_probability.npy", prior_prob)


def smooth_color_prior(sigma=5, data_dir=abs_data_dir, file="prior_probability.npy"):
    # """
    # Smooth color probability with a gaussian window
    # :param sigma: gaussian parameter
    # """
    prior_prob = np.load(os.path.join(data_dir, file))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    # create 1000 evenly spaced numbers [0,312]
    x_coord = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    y_coord = f(x_coord)
    # 2000 points in the window
    # Page 6 @ paper
    window = gaussian(2000, sigma)
    smoothed = convolve(y_coord, window / window.sum(), mode='same')
    fout = interp1d(x_coord, smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    np.save(os.path.join(data_dir, "prior_prob_smoothed.npy"), prior_prob_smoothed)


def compute_prior_factor(gamma=0.5, data_dir=abs_data_dir):
    # """
    # Calculate final weights from the smoothed probability distribution
    # :gamma: relationship factor between uniform probability and smoothed
    # :data_dir: directory containing smoothed probability values
    # """
    prior_prob_smoothed = np.load(os.path.join(data_dir, "prior_prob_smoothed.npy"))

    uni_probs = np.ones_like(prior_prob_smoothed)
    uni_probs = uni_probs / np.sum(1.0 * uni_probs)

    prior_factor = ((1 - gamma) * prior_prob_smoothed) + gamma * uni_probs
    prior_factor = np.power(prior_factor, -1)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    np.save(os.path.join(data_dir, "prior_factor.npy"), prior_factor)


def main():
    # --------------------------------------------------
    # # Optional: How to calculate prior probability
    # size = 64
    # # Load the sample of images
    # images_ab = load_data(size)
    # # Calculate prior probability of color
    # compute_color_prior(images_ab)
    # smooth_color_prior(file="prior_probability.npy")
    # compute_prior_factor()
    # --------------------------------------------------
    smooth_color_prior(file="zhang_probs.npy")
    compute_prior_factor()
    print("Calculated color priors. Exiting...")


if __name__ == '__main__':
    main()
