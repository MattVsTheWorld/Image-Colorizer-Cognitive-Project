import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from src.Old.preprocessing import ImageLoader


def loss_l3(found, truth):
    return tf.divide((tf.pow(tf.abs(tf.subtract(truth, found)), 3)), 3)


def main():
    # --- Parameters ---
    image_size = 256
    img_format = 'jpeg'
    # ------------------
    # --- Tensorflow warning settings ---
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # --------------------------------------
    loader = ImageLoader(image_size, img_format)
    train_images, train_labels = loader.load_folder(os.pardir
                                                    + '/imagenet/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/train/n01828970',
                                                    mp=True)
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    train_images = train_images.reshape((-1, image_size, image_size, 1))
    # train_labels = train_labels.reshape((-1, 256, 256, 3))

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # model.summary()
    model.compile(optimizer='adam',
                  loss=loss_l3)
    model.fit(train_images, train_labels, epochs=50)

    # /imagenet/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/train/n01828970/n01828970_62.jpeg
    img_bw, img_color = loader.load_img(os.pardir + '/imagenet/ILSVRC2017_CLS-LOC/ILSVRC/Data/'
                                                    'CLS-LOC/train/n01828970/n01828970_62.jpeg')
    cv2.imshow('lul', cv2.cvtColor(img_color, cv2.COLOR_Lab2BGR))
    cv2.waitKey()
    img_bw = np.expand_dims(img_bw, axis=0)
    img_bw = np.expand_dims(img_bw, axis=3)
    result = model.predict(img_bw)
    cv2.imshow('lul', cv2.cvtColor(result[0], cv2.COLOR_Lab2BGR))
    cv2.waitKey()


if __name__ == '__main__':
    main()
