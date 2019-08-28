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
from keras import metrics

from src.config import patience, epochs, batch_size, learning_rate, percentage_training
from src.data_generator import train_gen, valid_gen
from src.model import build_model
from src.image_pickler import image_unpickler

import numpy as np
import tensorflow as tf


def categorical_crossentropy_color(y_true, y_pred):
    q = 313

    # # y_true/pred is a distribution of probabilities of colors (313) for each pixel * each image
    # print("BALLAN")
    # y_true = keras.backend.print_tensor(y_true, message='\ny_true = ', summarize=313*8*8)
    # y_pred = keras.backend.print_tensor(y_pred, message='\ny_pred = ', summarize=313*8*8)
    # # shape = keras.backend.print_tensor(keras.backend.shape(y_true), message='\nshape = ', summarize=313)
    # print("SAD")

    y_true = keras.backend.reshape(y_true, (-1, q))
    y_pred = keras.backend.reshape(y_pred, (-1, q))
    idx_max = keras.backend.argmax(y_true, axis=1)

    prior_factor = np.load(os.path.join('data/', "prior_factor.npy")).astype(np.float32)
    weights = keras.backend.gather(prior_factor, idx_max)
    weights = keras.backend.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = keras.backend.categorical_crossentropy(y_pred, y_true)
    cross_ent = keras.backend.mean(cross_ent, axis=-1)
    # cross_ent = keras.backend.sum(cross_ent, axis=-1)

    return cross_ent


def save_model_cloud(model, job_dir, name='model'):
    filename = name + '.h5'
    model.save(filename)
    with file_io.FileIO(filename, mode='r') as inputFile:
        with file_io.FileIO(job_dir + '/' + filename, mode='w+') as outFile:
            outFile.write(inputFile.read())


def main():

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]
    checkpoint_models_path = 'models/'

    # Callbacks
    # gs://cs-b-logs'
    tensor_board = keras.callbacks.TensorBoard(log_dir='gs://cs-b-logs', histogram_freq=0, write_graph=True, write_images=True)
    # Save model (model.epoch.loss)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True, period=20)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    # save_model = LambdaCallback(on_epoch_end=lambda epoch, logs: save_model_cloud(epoch, 25))

    new_model = build_model()

    # if pretrained_path is not None:
    #     new_model.load_weights(pretrained_path)

    # TODO: Adam (doesn't work)
    # Optimizer
    # def euclidean_distance_loss(y_true, y_pred):
    #     """
    #     Euclidean distance loss
    #     https://en.wikipedia.org/wiki/Euclidean_distance
    #     :param y_true: TensorFlow/Theano tensor
    #     :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    #     :return: float
    #     """
    #     return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_pred - y_true), axis=-1))

    # --- SGD ---
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True, clipnorm=5.)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # --- Adam ---
    # opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # new_model.compile(optimizer=opt, loss=categorical_crossentropy_color)
    # ------------
    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    images = image_unpickler('images.pickle')

    # Read number of training and validation samples
    num_train_samples = floor(len(images) * percentage_training)
    num_valid_samples = floor(len(images) * (1 - percentage_training))

    # Start/resume training


    new_model.fit_generator(train_gen(images),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(images),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=8
                            )


if __name__ == '__main__':
    main()

# '''
# export JOB_NAME="test_job"
# export BUCKET_NAME=cs-b-bucket
# export CLOUD_CONFIG=src/cloudml-gpu.yaml
# export JOB_DIR=gs://cs-b-bucket/jobs/$JOB_NAME
# export MODULE=trainer.cloud._trainer
# export PACKAGE_PATH=./src
# export REGION=europe-west6
# export RUNTIME=1.2
# export TRAIN_FILE=gs://images_data/images.pickle
#
# gcloud ml-engine jobs submit training test_job --job-dir gs://cs-b-job-dir --runtime-version 1.2 --module-name trainer.cloud._trainer --package-path C:\Users\tomlo\Desktop\Cognitive-Project --region europe-west1 --config=src\cloudml-gpu.yaml --packages gs://images_data/images.pickle --module-name test_job
# '''
