import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import patience, epochs, batch_size, learning_rate, imgs_dir, checkpoint_models_path, save_period, fmt
from data_generator import train_gen, valid_gen, split_data
from model import build_model
from _class_rebalancing.color_loss import categorical_crossentropy_color


def main():
    # ------------------------------------------------------
    # Run training. Parameters specified in config
    # ------------------------------------------------------
    # Create tensorboard logs
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    # Save models. Models are saved only if improved; check is made every 'save_period' epochs
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss',
                                       verbose=1, save_best_only=True, period=save_period)
    # Stop training if validation loss does not improve after 'patience' epochs
    early_stop = EarlyStopping('val_loss', patience=patience)
    # Reduce learning rate if validation loss has stopped improving
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    # ------------------------------------------------------
    # ---------------------- Optimizer ---------------------
    # ------------------------------------------------------
    new_model = build_model(batchnorm=False)
    opt = keras.optimizers.Adam(lr=learning_rate)   # , decay=0.001)
    new_model.compile(optimizer=opt, loss=categorical_crossentropy_color(precalc=False))

    # -----------
    # sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True, clipnorm=5.)
    # new_model.compile(optimizer=opt, loss=euclidean_distance_loss)
    # opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # new_model.compile(loss=categorical_crossentropy_color(), optimizer=opt)
    # ,metrics=[metrics.categorical_accuracy])

    # ------------
    print(new_model.summary())
    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
    split_data(imgs_dir, fmt)
    # Read number of training and validation samples
    with open('image_names/train_num.txt', 'r') as f:
        num_train_samples = int(f.read())
    with open('image_names/valid_num.txt', 'r') as f:
        num_valid_samples = int(f.read())

    # Start training
    new_model.fit_generator(train_gen(imgs_dir),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(imgs_dir),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            # workers=8
                            )


if __name__ == '__main__':
    main()
