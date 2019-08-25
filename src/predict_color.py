# import the necessary packages
import os
from glob import glob
import random
from numpy.core._multiarray_umath import ndarray

import cv2
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras.backend as K
sys.stderr = stderr
import numpy as np
import sklearn.neighbors as nn

from src.config import img_rows, img_cols, data_dir, T, imgs_dir
from src.model import build_model


def main():
    channel: int = 3
    epsilon: float = 1e-8

    # Pick the latest weights. Only improved models are saved (latest = best loss)
    model_weights_path: str = max(glob('models/*.hdf5'), key=os.path.getctime)
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    image_folder: str = os.pardir + imgs_dir
    names_file: str = 'image_names/valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()
    # Pick 10 samples from validation set
    samples = random.sample(names, 10)

    height: int
    width: int
    height, width = img_rows // 4, img_cols // 4

    # Load the array of quantized ab value
    q_ab: ndarray = np.load(os.path.join(data_dir, "lab_gamut.npy"))
    nb_q: int = q_ab.shape[0]

    # Fit a NN to q_ab
    nn_finder = nn.NearestNeighbors(algorithm='ball_tree').fit(q_ab)

    # Clear folder
    for the_file in os.listdir('output_images'):
        file_path = os.path.join('output_images', the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr: ndarray = cv2.resize(cv2.imread(filename), (img_rows, img_cols), cv2.INTER_AREA)
        gray: ndarray = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (img_rows, img_cols), cv2.INTER_AREA)

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        # Split into respective channels
        L: ndarray = lab[:, :, 0]
        a: ndarray = lab[:, :, 1]
        b: ndarray = lab[:, :, 2]
        x_test: ndarray = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        # --- Prediction ---
        X_colorized: ndarray = model.predict(x_test)
        X_colorized = X_colorized.reshape((height * width, nb_q))

        # Reweight probabilities; epsilon avoids 0/NaN errors
        # Formula (5) @paper
        X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted; take all a/b values from color space
        q_a = q_ab[:, 0].reshape((1, 313))
        q_b = q_ab[:, 1].reshape((1, 313))

        # Add the predicted colors
        # axis 1 = colums; sum all values in rows
        X_a = np.sum(X_colorized * q_a, 1).reshape((height, width))
        X_b = np.sum(X_colorized * q_b, 1).reshape((height, width))

        X_a = cv2.resize(X_a, (img_rows, img_cols), cv2.INTER_CUBIC)
        X_b = cv2.resize(X_b, (img_rows, img_cols), cv2.INTER_CUBIC)

        # Before: -90 <=a<= 100, -110 <=b<= 110
        # After: 38 <=a<= 228, 18 <=b<= 238
        X_a = X_a + 128
        X_b = X_b + 128

        out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b

        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

        out_bgr = out_bgr.astype(np.uint8)

        if not os.path.exists('output_images'):
            os.makedirs('output_images')

        cv2.imwrite('output_images/{}_bw.png'.format(i), gray)
        cv2.imwrite('output_images/{}_gt.png'.format(i), bgr)
        cv2.imwrite('output_images/{}_out.png'.format(i), out_bgr)

    K.clear_session()


if __name__ == '__main__':
    main()
