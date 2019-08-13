import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from src.preprocessing import ImageLoader


def loss_l3(found, truth):
    return tf.divide((tf.pow(tf.abs(tf.subtract(truth, found)), 3)), 3)


def main():
    IMG_SIZE = 256
    # Allows INFO logs to be printed
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # Set verbosity of warnings allowed (higher number, less warnings)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # --------------------------------------
    loader = ImageLoader(IMG_SIZE)
    train_images, train_labels = loader.load_folder(os.pardir + '/test_imgs/a_subfolder')
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    train_images = train_images.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    # train_labels = train_labels.reshape((-1, 256, 256, 3))

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # model.summary()
    model.compile(optimizer='adam',
                  loss=loss_l3)
    model.fit(train_images, train_labels, epochs=200)

    img, kappa = loader.load_img(os.pardir + '/test_imgs/a_subfolder/twitch.jpg')
    cv2.imshow('lul', cv2.cvtColor(kappa, cv2.COLOR_Lab2BGR))
    cv2.waitKey()
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    result = model.predict(img)
    cv2.imshow('lul', cv2.cvtColor(result[0], cv2.COLOR_Lab2BGR))
    cv2.waitKey()


if __name__ == '__main__':
    main()
