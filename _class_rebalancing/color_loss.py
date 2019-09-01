from keras.backend import reshape, gather, argmax, categorical_crossentropy, mean, sum
import numpy as np
import os
from config import num_colors


def categorical_crossentropy_color(y_true, y_pred, precalc=True):
    # # y_true/pred is a distribution of probabilities of colors (313) for each pixel * each image
    # y_true = keras.backend.print_tensor(y_true, message='\ny_true = ', summarize=313*8*8)
    # y_pred = keras.backend.print_tensor(y_pred, message='\ny_pred = ', summarize=313*8*8)
    # # shape = keras.backend.print_tensor(keras.backend.shape(y_true), message='\nshape = ', summarize=313)

    y_true = reshape(y_true, (-1, num_colors))
    y_pred = reshape(y_pred, (-1, num_colors))
    idx_max = argmax(y_true, axis=1)
    # Use own prior factor, calculated in class_rebalancing
    if not precalc:
        prior_factor = np.load(os.path.join('data/', "prior_factor.npy")).astype(np.float32)
    # Use prior factor as calculated by Zhang et al in their paper
    else:
        prior_factor = np.load(os.path.join('data/', "prior_probs_zhang.npy")).astype(np.float32)
    weights = gather(prior_factor, idx_max)
    weights = reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = categorical_crossentropy(y_pred, y_true)
    cross_ent = mean(cross_ent, axis=-1)
    # cross_ent = sum(cross_ent, axis=-1)

    return cross_ent
