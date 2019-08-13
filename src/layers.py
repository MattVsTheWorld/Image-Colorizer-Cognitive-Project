import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class CustomConv(keras.layers.Layer):
    def __init__(self, units=256):
        # TODO: fix #units
        super(CustomConv, self).__init__()
        self.units = units

    def build(self, input_shape):
        # TODO: implement
        print("build weights")

    def call(self, inputs, **kwargs):
        # TODO: implement
        return 0
        


def main():
    # Allows INFO logs to be printed
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # Set verbosity of warnings allowed (higher number, less warnings)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    main()
