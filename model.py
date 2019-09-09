import os
import sys
# Silence unwanted warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model

from config import img_rows, img_cols, num_colors, kernel_size
from config import layer_init as kernel_init


def build_model():
    tf.logging.set_verbosity(tf.logging.ERROR)
    # Initialize l2 regulator from keras
    l2_reg = l2(1e-3)

    # Input image of specified shape (black and white)
    # Input data is a sizexsize tensor containing l values
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 1 ----------------------------------
    # ----------------------------------------------------------------------------
    # 64 (output channels), 3x3 kernel
    x = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv1_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(input_tensor)
    # Spacial resolution of output = 224
    x = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv1_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 112
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 2 ----------------------------------
    # ----------------------------------------------------------------------------
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv2_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 112
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv2_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 56
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 3 ----------------------------------
    # ----------------------------------------------------------------------------
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 56
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 56
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 4 ----------------------------------
    # ----------------------------------------------------------------------------
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg,  strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg,  strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg,  strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 5 ----------------------------------
    # ----------------------------------------------------------------------------
    # Notice dilated convolution
    # Dilated convolution is a basic convolution only applied to the input volume with defined gaps
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 6 ----------------------------------
    # ----------------------------------------------------------------------------
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 7 ----------------------------------
    # ----------------------------------------------------------------------------
    # No more dilation
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # ---------------------------------- Conv 8 ----------------------------------
    # ----------------------------------------------------------------------------
    # UpSample before convolution
    x = UpSampling2D(size=(2, 2))(x)
    # Spacial resolution of output = 56
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=1, name='conv8_1', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)

    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=1, name='conv8_2', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=1, name='conv8_3', kernel_initializer=kernel_init,
               kernel_regularizer=l2_reg, strides=(1, 1))(x)

    outputs = Conv2D(num_colors, (1, 1), activation='softmax', padding='same',
                     dilation_rate=1, name='conv8_313')(x)
    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model


def main():

    encoder_decoder = build_model()
    # Print specifics
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)
    K.clear_session()


if __name__ == '__main__':
    main()
