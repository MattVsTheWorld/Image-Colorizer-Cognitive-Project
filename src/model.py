import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from keras.regularizers import l2
# from keras.utils import multi_gpu_model
from keras.utils import plot_model

from src.config import img_rows, img_cols, num_colors, kernel_size


def build_model():
    # Initialize l2 regulator from keras
    l2_reg = l2(1e-3)

    # Input image of specified shape (black and white)
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # 64 (output channels), 3x3 kernel
    # he normal = truncated normal distribution centered on 0
    # ----------------------------------------------------------------------------
    x = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv1_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(input_tensor)
    # Spacial resolution of output = 224
    x = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv1_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 112
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv2_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 112
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv2_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 56
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 56
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 56
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv3_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv4_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # Notice dilated convolution
    # Dilated convolution is a basic convolution only applied to the input volume with defined gaps
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv5_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
               dilation_rate=2, name='conv6_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    # No more dilation
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv7_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # Spacial resolution of output = 28
    x = BatchNormalization()(x)
    # ----------------------------------------------------------------------------
    x = UpSampling2D(size=(2, 2))(x)
    # Spacial resolution of output = 56
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv8_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(.5, .5))(x)
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv8_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    x = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
               name='conv8_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(1, 1))(x)
    # 1x1 conv and cross-entropy loss layer
    # TODO: check softmax
    outputs = Conv2D(num_colors, (1, 1), activation='softmax', padding='same', name='pred')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model


def main():
    # Build the model on the CPU
    with tf.device("/cpu:0"):
        encoder_decoder = build_model()
    # Print specifics
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)

    # # Possibility of multi-gpu model
    # parallel_model = multi_gpu_model(encoder_decoder, gpus=None)
    # print(parallel_model.summary())
    # plot_model(parallel_model, to_file='parallel_model.svg', show_layer_names=True, show_shapes=True)

    # try:
    #     model = multi_gpu_model(model, cpu_relocation=True)
    #     print("Training using multiple GPUs..")
    # except:
    #     print("Training using single GPU or CPU..")

    # Destroys the current TF graph and creates a new one
    K.clear_session()


if __name__ == '__main__':
    main()
