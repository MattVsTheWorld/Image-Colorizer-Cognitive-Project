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
from config import img_rows, img_cols, data_dir, T, imgs_dir, fmt
from model import build_model
from utils import clear_folder
import tensorflow as tf


def colorize(model, x_test, height, width, nb_q, q_ab, lab):
    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    # ------------------------------------------------------
    # --------------------- Prediction ---------------------
    # ------------------------------------------------------
    X_colorized = model.predict(x_test)
    X_colorized = X_colorized.reshape((height * width, nb_q))

    # Reweight probabilities; epsilon avoids 0/NaN errors
    # Formula (5) @paper
    epsilon = 1e-8
    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted; take all a/b values from color space
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    # Add the predicted colors
    # axis 1 = colums; sum all values in rows
    X_a = np.sum(X_colorized * q_a, 1).reshape((height, width))
    X_b = np.sum(X_colorized * q_b, 1).reshape((height, width))

    X_a = cv2.resize(X_a, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    X_b = cv2.resize(X_b, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)

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
    return out_bgr


def main():
    # ------------------------------------------------------
    # Run predictor on some validation images
    # ------------------------------------------------------
    # Toggle to force prediction on cpu (if gpu is busy)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    # Latest model is loaded, as only improvements are saved
    model_1_weights_path = max(glob('models/*.hdf5'), key=os.path.getctime)
    model_2_weights_path = min(glob('models/*.hdf5'), key=os.path.getctime)

    model_1 = build_model(True)
    model_2 = build_model(False)
    model_1.load_weights(model_1_weights_path)
    model_2.load_weights(model_2_weights_path)

    print(model_1.summary())

    names = [f for f in os.listdir(imgs_dir
                                   ) if f.lower().endswith(fmt)]

    # Pick 10 samples from validation set
    # samples = random.sample(names, 10)

    height, width = img_rows // 4, img_cols // 4

    # Load the array of quantized ab value
    q_ab = np.load(os.path.join(data_dir, "lab_gamut.npy"))
    nb_q = q_ab.shape[0]

    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    clear_folder('output_images')

    print("----------------------------------------\n"
          "Prediction 1 based on " + model_1_weights_path[7:] + "\n"
          "Prediction 2 based on " + model_2_weights_path[7:] + "\n"
          "----------------------------------------")

    for i in range(len(names)):
        image_name = names[i]
        filename = os.path.join(imgs_dir, image_name)
        print('Processing image: {}'.format(filename[16:]))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv2.imread(filename)
        gray = cv2.imread(filename, 0)
        bgr = cv2.resize(bgr, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        out_bgr_1 = colorize(model_1, x_test, height, width, nb_q, q_ab, lab)
        out_bgr_2 = colorize(model_2, x_test, height, width, nb_q, q_ab, lab)

        # cv2.imwrite('output_images/{}_bw.png'.format(i), gray)
        cv2.imwrite('output_images/{}_gt.png'.format(i), bgr)
        cv2.imwrite('output_images/{}_out_1.png'.format(i), out_bgr_1)
        cv2.imwrite('output_images/{}_out_2.png'.format(i), out_bgr_2)

    K.clear_session()


if __name__ == '__main__':
    main()

