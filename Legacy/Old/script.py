# ---- Imports ----
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy.core._multiarray_umath import ndarray
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2
import os
from scipy import stats
from skimage import color
import tensorflow as tf
import random
from tqdm import tqdm
# --- /////// ----


# Softmax activation function (There is also the default one but is better to have it written for T R A S P A R E N C Y)
def softmax(z):
    return tf.exp(z) / tf.reduce_sum(tf.exp(z))


# Gaussian filter definition (There is also the default one but is better to have it written for T R A S P A R E N C Y)
def gaussian_filter(kernel_len: int = 21, sigma: int = 3) -> ndarray:
    """
    :param kernel_len: gaussian parameter
    :param sigma: gaussian parameter
    :return: a 2D Gaussian kernel array
    """
    interval = (2 * sigma + 1.) / kernel_len
    x = np.linspace(- sigma - interval / 2., sigma + interval / 2., kernel_len + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel: ndarray(dtype=float, shape=(kernel_len, kernel_len)) = kernel_raw / kernel_raw.sum()
    return kernel


# Function to convert the NxNx2 distr. in a soft-encoding distribution NxNxQ
def lab2distr(image: ndarray, map_size: int,
              eq_map: ndarray(dtype=float, shape=(256, 256)),
              eq_map_small: ndarray(dtype=float, shape=(32, 32)),
              c1: ndarray, c2: ndarray, tiles_num: int) -> ndarray:
    # 2D Gaussian parameters
    kernel_len: int = 5
    sigma: int = 2
    gauss_filter: ndarray(dtype=float, shape=(kernel_len, kernel_len)) = gaussian_filter(kernel_len, sigma)
    # ndarray of zeros and same shape as image, but it's N x N x tiles_num rather than x 2
    result: ndarray(dtype=float, shape=image.shape) = np.zeros((image.shape[0], image.shape[0], tiles_num))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # create a temporary matrix with added kernel padding to the sides
            tmp: ndarray = np.zeros((eq_map_small.shape[0] + (kernel_len // 2) * 2,      # *2???  TODO: forse // 2 *2 è per assicurarsi che sia divisibile? forza il rounding
                                     eq_map_small.shape[1] + (kernel_len // 2) * 2))     # 5 // 2 = 2, 32 + 2*2 = 36
            # Get the a,b values that will be like coordinates
            # and shift the values to fit the matrix (-127 <= a,b <= 128.
            a: float    # a* of L*a*b*
            b: float    # b* of L*a*b*
            a, b = image[i, j, :]
            a: int = (int(a) + map_size // 2) - 1  # TODO: io il -1 lo metto, mi sembra giusto
            b: int = (int(b) + map_size // 2) - 1

            color_class: float = eq_map[a, b]
            # if non weight is related, error (not 100% sure if needed)
            if color_class < 1:
                print("Error in color_class creation")

            # find the relative center coordinates
            center_x_arr: ndarray(dtype=int, shape=(1,))
            center_y_arr: ndarray(dtype=int, shape=(1,))
            center_x_arr, center_y_arr = np.where(eq_map_small == color_class)
            # stores as 1-element arrays
            center_x: int = center_x_arr[0] + kernel_len // 2
            center_y: int = center_y_arr[0] + kernel_len // 2

            # apply the gaussian filter
            tmp[center_x - kernel_len // 2: center_x + kernel_len // 2 + 1,
                center_y - kernel_len // 2: center_y + kernel_len // 2 + 1] = gauss_filter
            # Remove the borders I added before
            tmp = tmp[kernel_len // 2: -1 * (kernel_len // 2), kernel_len // 2: -1 * (kernel_len // 2)]
            # Now I flatten it and take the significant bins.
            result[i, j, :] = tmp[c1, c2]
    return result


# function to convert from NxNxQ to NxNx3 (original image).
def distr2lab(bwimage: ndarray, win_map: int, map_size: int,
              eq_map_small: ndarray(dtype=float, shape=(32, 32)),
              c1: ndarray, c2: ndarray) -> ndarray:
    # Create an empty matrix for channels a, b
    image: ndarray = np.zeros((bwimage.shape[0], bwimage.shape[1], 2))
    for i in range(bwimage.shape[0]):
        for j in range(bwimage.shape[1]):
            # Take values at coordinates, throughout tiles_num dimensions //TODO: giusto?
            result_2: ndarray(dtype=float, shape=(1,)) = bwimage[i, j, :]

            # TODO: mysterious formula
            # res2=np.exp(np.log(res2)/T)/(np.sum(np.exp(np.log(res2)/T)))
            # (suspect, reasonable with high probability, investigate)

            matrix: ndarray(dtype=float, shape=(32, 32)) = np.zeros_like(eq_map_small)
            # Put the distribution back to the original colorspace
            matrix[c1, c2] = result_2

            c1, c2 = np.where(matrix != 0)
            probabilities: ndarray = matrix[matrix != 0]
            color_x: float = np.sum(c1 * (probabilities * 10)) / np.sum(probabilities * 10)    # *10?
            color_y: float = np.sum(c2 * (probabilities * 10)) / np.sum(probabilities * 10)

            color_x = color_x * win_map - map_size // 2   # why *WinMap ? / = cause 32*32
            color_y = color_y * win_map - map_size // 2
            # put color_x, color_y at coordinates i, j
            image[i, j] = [color_x, color_y]

    return image


# function to map weights to final values (kernel excluded)
# TODO: some types missing
def map_weights(batch, win_map: int,
                final_weights: ndarray(dtype=float, shape=(32, 32))) -> ndarray:
    result: ndarray = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2]))
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            for k in range(batch.shape[2]):
                # traspone la tabella
                ind1: int = (int(batch[i, j, k, 0]) + 127) // win_map
                ind2: int = (int(batch[i, j, k, 1]) + 127) // win_map
                result[i, j, k] = final_weights[ind1, ind2]

    return result


# TODO: finish typing
def getModel(tiles_num: int, eps: float = 10e-9):
    X = tf.placeholder("float", [None, 32, 32, 1])
    Y = tf.placeholder("float", [None, 16, 16, tiles_num])
    ZW = tf.placeholder("float", [None, 16, 16])

    step = tiles_num // 8

    with tf.variable_scope("conv1") as scope:
        W = tf.get_variable("W", shape=[3, 3, 1, step],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", initializer=tf.zeros([step]))
        l = tf.nn.bias_add(
            tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID"), b)
        l_act = tf.nn.relu(l)

    for i in range(2, 8):
        with tf.variable_scope("conv" + str(i)) as scope:
            W = tf.get_variable("W", shape=[3, 3, step * (i - 1), step * i],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", initializer=tf.zeros([step * i]))
            l = tf.nn.bias_add(
                tf.nn.conv2d(l_act, W, strides=[1, 1, 1, 1], padding="VALID"), b)
            l_act = tf.nn.relu(l)

    with tf.variable_scope("conv8") as scope:
        W = tf.get_variable("W", shape=[3, 3, step * 7, tiles_num],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", initializer=tf.zeros([tiles_num]))
        output = tf.nn.bias_add(
            tf.nn.conv2d(l, W, strides=[1, 1, 1, 1], padding="VALID"), b)
    # l_act = tf.nn.softmax(l)

    pred = tf.nn.softmax(output)
    # I should add the weights
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))

    # I guess i should reduce it
    # cost = tf.reduce_mean(ZW*-tf.reduce_sum(Y*tf.log(pred),3)) #16,16,16
    # I get nans
    # When I am getting nans it is usually either of the three:
    # - batch size too small (in your case then just 1)
    # - log(0) somewhere
    # - learning rate too high and uncapped gradients
    cost = tf.reduce_sum(ZW * -tf.reduce_sum(Y * tf.log(pred + eps), 3))  # 16,16,16

    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    return (X, Y, ZW), train_step, cost, pred


# TODO: finish typing
def run(tiles_num: int, win_map: int, map_size: int,
        eq_map: ndarray(dtype=float, shape=(256, 256)),
        eq_map_small: ndarray(dtype=float, shape=(32, 32)),
        c1: ndarray, c2: ndarray,
        final_weights: ndarray(dtype=float, shape=(32, 32)),
        num_epochs: int = 10, batch_size: int = 32):
    # X, Y, ZW are tf.placeholders
    (X, Y, ZW), train_step, cost, pred = getModel(tiles_num)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        images: List[ndarray] = []
        for filename in tqdm(os.listdir(os.pardir + "/test_imgs/bird")):
            im: ndarray = cv2.cvtColor(cv2.imread(os.pardir + "/test_imgs/bird/" + filename), cv2.COLOR_BGR2RGB)
            im_lab: ndarray(dtype=float, shape=(256, 256, 3)) = color.rgb2lab(cv2.resize(im, (256, 256)))
            images.append(im_lab)

        iters: List[Tuple[int, int, int]] = [(x, y, z) for z in range(0, 256, 32)
                                             for y in range(0, 256, 32)
                                             for x in range(len(images))]

        for ep in range(num_epochs):
            random.shuffle(iters)
            cc = 0
            print("Epoch: {0}.".format(ep))
            for batch in range(0, len(images), batch_size):
                cc += 1
                input_batch = np.zeros((batch_size, 32, 32, 1))
                labels_batch = np.zeros((batch_size, 16, 16, tiles_num))
                colors_batch = np.zeros((batch_size, 16, 16, 2))
                for index in range(batch_size):
                    imID, i, j = iters[index]
                    piece = images[imID][i:i + 32, j:j + 32, :]
                    input_batch[index, :, :, 0] = piece[:, :, 0]
                    labels_batch[index, :, :, :] = np.reshape(lab2distr(piece[8:-8, 8:-8, 1:], map_size,
                                                                        eq_map, eq_map_small, c1, c2,
                                                                        tiles_num), (1, 16, 16, tiles_num))
                    colors_batch[index, :, :, :] = np.reshape(piece[8:-8, 8:-8, 1:], (1, 16, 16, 2))
                # print(colors_batch)
                [_, c] = sess.run([train_step, cost],
                                  feed_dict={X: input_batch, Y: labels_batch, ZW: map_weights(colors_batch, win_map, final_weights)})
                print(c)

        saver.save(sess, "test-model-good2")
        # Testing
        for kk in range(10):
            ind = int(random.uniform(0, len(images)))
            res = np.zeros((256, 256, 3))
            res[:, :, 0] = images[ind][:, :, 0]
            lala = []
            lolo = []
            for i in range(8, 256 - 8, 16):
                for j in range(8, 256 - 8, 16):
                    inp = np.reshape(images[ind][i - 8:i + 24, j - 8:j + 24, 0], (1, 32, 32, 1))

                    [p] = sess.run([pred], feed_dict={X: inp})
                    p = np.reshape(p, (16, 16, tiles_num))
                    lala.append(p)
                    lolo.append(inp)
                    dp = distr2lab(p, win_map, map_size, eq_map_small, c1, c2)

                    res[i:i + 16, j:j + 16, 1:] = dp
                # print(dp.shape)

            res_converted = color.lab2rgb(res)
            res2 = np.zeros_like(res)
            res2[:, :, 1:] = res[:, :, 1:]
            res2[:, :, 0] = 100
            res2 = color.lab2rgb(res2)

            plt.figure()
            plt.imshow(res_converted)
            plt.show()


def populate_maps(map_size, win_map, distr_map, eq_map, eq_map_small, p_tilde):
    """
    Populates distribution maps
    :param map_size: size of the original map, loaded from file
    :param win_map: width/height of the square window used to divide the distribution map
    :param distr_map: (256, 256) the distribution map, loaded from file
    :param eq_map: (256, 256) contains the indexes      TODO: better description
    :param eq_map_small:  (32, 32) indexes, but small   TODO: better description
    :param p_tilde: (32, 32) contains the sum of probabilities in the win_map sized windows
    :return: the number of tiles and the filled in maps
    """
    idx = 0
    for i in range(0, map_size, win_map):
        for j in range(0, map_size, win_map):
            # sum the values in the windows i to i+win_map, j to j+win_map
            value_sum = np.sum(distr_map[i:i + win_map, j:j + win_map])
            # If the region has no value whatsoever just ignore it
            if value_sum > 0:
                # Save the region index in both tables and the corresponding probability
                idx += 1
                # from coordinates [i to i + win_map, j to j + win_map]
                eq_map[i:i + win_map, j:j + win_map] = idx
                eq_map_small[i // win_map, j // win_map] = idx
                p_tilde[i // win_map, j // win_map] = value_sum

    return idx, eq_map, eq_map_small, p_tilde


def main():
    print("Script started...")
    # ---- (1) Setup ----
    # Clears the default graph stack and resets the global default graph.
    tf.compat.v1.reset_default_graph()
    # Distributed map showing where most of the color values are found, loaded from file
    distr_map: ndarray(dtype=float, shape=(256, 256)) = np.load("color_distribution.npy")  # 256x256
    map_size: int = distr_map.shape[0]
    # Width and height of the window used to divide the distrMap equally
    win_map: int = 8  # 256/8 -> 32. 32*32 = 1024 tiles.
    # Small map: contains the tiles already reduced
    eq_map_small: ndarray(dtype=float, shape=(32, 32)) = np.zeros((map_size // win_map, map_size // win_map))
    # Contains the whole distributed map
    eq_map: ndarray(dtype=float, shape=(256, 256)) = np.zeros_like(distr_map)
    # Probability distribution used to weight the cross-entropy during the opt.
    p_tilde: ndarray(dtype=float, shape=(32, 32)) = eq_map_small.copy()
    # Temperature: 0 -> one-hot vector, 1 -> more mean/saturated
    T: float = 0.01
    # ---- ///// ----

    # ---- (2) Set up maps ----
    # Number of tiles
    tiles_num: int
    # Number of discrete regions of LAB colorspace
    tiles_num, eq_map, eq_map_small, p_tilde = populate_maps(map_size, win_map, distr_map,
                                                             eq_map, eq_map_small, p_tilde)
    # ---- //////// ----

    # ---- (3) Set up weights for colors ----
    # Coordinates where the significant bins can be found (c1: array of row indices, c2: array of column indices)
    c1: ndarray     # righe
    c2: ndarray     # colonne
    c1, c2 = np.where(eq_map_small != 0)
    # Transform the probability table into a vector of non-zero element (flatten)
    # e.g. 381 coordinate: diventa un array di 381 elementi
    p_tilde: ndarray = p_tilde[c1, c2]
    # Paper formula coefficient (suggested value)
    lambda_val: float = 0.5
    # Operazione su p_tilde; stessa forma
    weights: ndarray(dtype=float, shape=(32, 32)) = 1 / ((1 - lambda_val) * p_tilde + (lambda_val / tiles_num))
    # Normalize such that np.sum(Weights*P_tilde)==1
    weights = weights / np.sum(p_tilde * weights)

    # Create a matrix of the usual 32x32 size
    final_weights: ndarray(dtype=float, shape=(32, 32)) = np.zeros_like(eq_map_small)
    # put the weights at the corresponding coordinates
    final_weights[c1, c2]: ndarray(dtype=float, shape=(32, 32)) = weights
    # ---- //////////////////// ----

    # ---- (4) Run ----
    num_epochs: int = 10
    batch_size: int = 32

    run(tiles_num, win_map, map_size, eq_map,
        eq_map_small, c1, c2, final_weights, num_epochs, batch_size)
    # ---- ///// ----


if __name__ == '__main__':
    main()
