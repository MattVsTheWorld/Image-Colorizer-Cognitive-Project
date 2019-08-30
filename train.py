import argparse
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from math import floor
from tensorflow.python.lib.io import file_io

from config import patience, epochs, batch_size, learning_rate, percentage_training, imgs_dir
from data_generator import train_gen, valid_gen, split_data
from model import build_model
import tensorflow as tf
import numpy as np
from keras import backend as K

def main():

    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True, period=20)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    # print(K.tensorflow_backend._get_available_gpus())
    #     #
    #     # with tf.device("/device:GPU:0"):
    #     #     new_model = build_model()
    new_model = build_model()
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True, clipnorm=5.)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # -----------
    # opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # new_model.compile(optimizer=opt, loss=euclidean_distance_loss)
    # opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # new_model.compile(loss=categorical_crossentropy_color(), optimizer=opt)
    #  TODO: Test
    # ,metrics=[metrics.categorical_accuracy])

    # adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
    # new_model.compile(optimizer=adam, loss=categorical_crossentropy_color)
    # Print model stats#
    # ------------
    print(new_model.summary())
    image_folder = imgs_dir
    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
    split_data(image_folder, '.jpeg')
    # Read number of training and validation samples
    with open('image_names/train_num.txt', 'r') as f:
        num_train_samples = int(f.read())
    with open('image_names/valid_num.txt', 'r') as f:
        num_valid_samples = int(f.read())

    # Start/resume training

    new_model.fit_generator(train_gen(image_folder[1:]),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(image_folder[1:]),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            # workers=8
                            )


if __name__ == '__main__':
    main()
