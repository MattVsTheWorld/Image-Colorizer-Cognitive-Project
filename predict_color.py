import os
from glob import glob
import cv2
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras.backend as K
sys.stderr = stderr
import numpy as np
from config import img_rows, img_cols, data_dir, T
from model import build_model
from utils import clear_folder
import tensorflow as tf


def colorize(model, x_test, height, width, nb_q, q_ab, lab):
    # ------------------------------------------------------
    # --------------------- Prediction ---------------------
    # ------------------------------------------------------
    images_colorized = model.predict(x_test)
    images_colorized = images_colorized.reshape((height * width, nb_q))
    # We now have an array of h*w with 313 axes. Each value corresponds to the probability that point has that (of the 313) colors

    # Reweight probabilities; epsilon avoids 0/NaN errors
    # Formula (5) @paper
    epsilon = 1e-8
    images_colorized = np.exp(np.log(images_colorized + epsilon) / T)
    images_colorized = images_colorized / np.sum(images_colorized, 1)[:, np.newaxis]

    # Reweighted; take all a/b values from color space
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    # -----------------------------------
    # Sum all color probabilities. The highest probability will determine the color
    # These "color weights" are summed so trainsitions from one color the the other are smoother
    images_a = np.sum(images_colorized * q_a, 1).reshape((height, width))
    images_b = np.sum(images_colorized * q_b, 1).reshape((height, width))

    images_a = cv2.resize(images_a, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    images_b = cv2.resize(images_b, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)

    images_a = images_a + 128
    images_b = images_b + 128

    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = lab[:, :, 0]
    out_lab[:, :, 1] = images_a
    out_lab[:, :, 2] = images_b

    out_lab = out_lab.astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

    out_bgr = out_bgr.astype(np.uint8)
    return out_bgr


def main():
    # ------------------------------------------------------
    # Run predictor on some validation images
    # ------------------------------------------------------
    # Toggle to force prediction on cpu (if gpu is busy)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    # Latest model is loaded, as only improvements are saved
    model_1_weights_path = max(glob('models/*.hdf5'), key=os.path.getctime)
    model_2_weights_path = min(glob('models/*.hdf5'), key=os.path.getctime)

    model_1 = build_model()
    model_2 = build_model()
    model_1.load_weights(model_1_weights_path)
    model_2.load_weights(model_2_weights_path)

    print(model_1.summary())

    predict_folder = 'test_images/misc'
    # predict_folder = 'alt_set'

    names = [f for f in os.listdir(predict_folder) if f.lower().endswith('.jpg')]
    names_jpeg = [f for f in os.listdir(predict_folder) if f.lower().endswith('.jpeg')]
    names_png = [f for f in os.listdir(predict_folder) if f.lower().endswith('.png')]
    names = names + names_jpeg + names_png
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
          "Prediction '1' based on " + model_1_weights_path + "\n"
          "Prediction '2' based on " + model_2_weights_path + "\n"
          "----------------------------------------")

    for i in range(len(names)):
        image_name = names[i]
        filename = os.path.join(predict_folder, image_name)
        print('Processing image: {}'.format(filename))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv2.imread(filename)
        gray = cv2.imread(filename, 0)
        bgr = cv2.resize(bgr, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        out_bgr_1 = colorize(model_1, x_test, height, width, nb_q, q_ab, lab)
        out_bgr_2 = colorize(model_2, x_test, height, width, nb_q, q_ab, lab)

        # cv2.imwrite('output_images/{}_bw.png'.format(i), gray)
        cv2.imwrite('output_images/{}_gt.png'.format(i), bgr)
        cv2.imwrite('output_images/{}_1.png'.format(i), out_bgr_1)
        cv2.imwrite('output_images/{}_2.png'.format(i), out_bgr_2)

    K.clear_session()


if __name__ == '__main__':
    main()

