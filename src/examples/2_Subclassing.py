import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 1 - Model subclassing
class MyModel(keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # DEFINIZIONE DEI LAYER
        self.dense_1 = keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense_2 = keras.layers.Dense(num_classes, activation=tf.sigmoid)

    def call(self, inputs):
        # Definisci i forward pass dei layer definiti in init
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # Metodo che deve necessita override se si usa il functional-style model
        # else non serve (mah)
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


# 2 - Custom layers
class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

        ''' WIP - To be continued'''

# Instanziazioni
# Instance the model
model = MyModel(num_classes=15)