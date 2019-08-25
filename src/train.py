import argparse
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import metrics

from src.config import patience, epochs, batch_size, learning_rate, imgs_dir
from src.data_generator import train_gen, valid_gen
from src.model import build_model

# TODO: TEST ---------------------------------------
import numpy as np
import tensorflow as tf


def categorical_crossentropy_color(y_true, y_pred):
    q = 313
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

    return cross_ent
# TODO: TEST ---------------------------------------


def main():

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    pretrained_path = args["pretrained"]
    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    # Save model (model.epoch.loss)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    new_model = build_model()

    # if pretrained_path is not None:
    #     new_model.load_weights(pretrained_path)

    # TODO: Adam (doesn't work)
    # Optimizer
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True, clipnorm=5.)
    new_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #  TODO: Test
    # ,metrics=[metrics.categorical_accuracy])

    # adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
    # new_model.compile(optimizer=adam, loss=categorical_crossentropy_color)
    # Print model stats
    print(new_model.summary())

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Read number of training and validation samples
    with open('image_names/train_num.txt', 'r') as f:
        num_train_samples = int(f.read())
    with open('image_names/valid_num.txt', 'r') as f:
        num_valid_samples = int(f.read())

    # Start/resume training
    image_folder: str = os.pardir + imgs_dir
    new_model.fit_generator(train_gen(image_folder),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(image_folder),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=8
                            )


if __name__ == '__main__':
    main()
