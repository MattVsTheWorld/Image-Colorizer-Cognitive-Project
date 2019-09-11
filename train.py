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
from color_loss import categorical_crossentropy_color


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
    new_model = build_model()
    opt = keras.optimizers.Adam(lr=learning_rate)
    new_model.compile(optimizer=opt, loss=categorical_crossentropy_color)

    # ------------
    print(new_model.summary())
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
