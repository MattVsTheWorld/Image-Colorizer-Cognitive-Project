from keras.backend import reshape, gather, argmax, categorical_crossentropy, mean
import numpy as np
import os
from config import num_colors


def categorical_crossentropy_color(y_true, y_pred):

    y_true = reshape(y_true, (-1, num_colors))
    y_pred = reshape(y_pred, (-1, num_colors))
    # take the most probable colour of the 5 nearest neighbours
    idx_max = argmax(y_true, axis=1)

    prior_factor = np.load(os.path.join('data/', "prior_factor.npy")).astype(np.float32)

    # Find weight of the corresponding color. Rarer colors have higher weight
    weights = gather(prior_factor, idx_max)
    weights = reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = categorical_crossentropy(y_true, y_pred)
    cross_ent = mean(cross_ent, axis=-1)

    return cross_ent
