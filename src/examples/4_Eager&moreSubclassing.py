from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# ------------------- REFS -------------------
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/guide/keras/custom_layers_and_models.ipynb#scrollTo=ejSYZGaP4CD6
# --------------------------------------------

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------- EAGER --------
# Eager execution, a loro dire, funziona bene col subclassing e layer custom
# Da mettere una sola volta all'inizio del programma
tf.enable_eager_execution()
# check
tf.executing_eagerly()
# ------------------------


# -------- More on custom layers --------
# 1 - Recap -
class Linear1(keras.layers.Layer):

    # Stato del layer; i suoi pesi/variabili
    def __init__(self, units=32, input_dim=32):
        super(Linear1, self).__init__()
        # le variabili w e b rappresentano lo stato del layer
        # Due modi di aggiungere un peso:
        # ---------------------
        # 1 - verbose
        '''
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        '''
        # 2 - quicker (uguali
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 # Se settato a false, i pesi non vengono considerati durante la backpropagation
                                 trainable=True)
        # ---------------------
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)
        # ---------------------

    # Funzione del layer; trasformazione da input ad output
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# 2 - Best practice: creare i pesi solo quando si sa lo shape dell'input
# (invece di inizalizzarli così)

class Linear2(keras.layers.Layer):

    def __init__(self, units=32):
        super(Linear2, self).__init__()
        self.units = units

    # Possiamo chiamare build in modo "lazy" quando conosciamo l'input
    # Piuttosto che inizializzare i pesi alla creazione
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
    # NOTA: una chiamata a __call__ chiama anche 'build' la prima volta che viene chiamato
    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b


x_ = tf.ones((2, 2))
linear_layer = Linear1(4, 2)
y = linear_layer(x_)
print(y)

linear_layer = Linear2(32)  # At instantiation, we don't know on what inputs this is going to get called
y = linear_layer(x_)         # The layer's weights are created dynamically the first time the layer is called
print(y)


# 3 - I layer sono componibili ricorsivamente
class MLPBlock(keras.layers.Layer):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear2(32)
        self.linear_2 = Linear2(32)
        self.linear_3 = Linear2(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))


# 4 - Si può creare un tensore di loss da usare durante il training
class ActivityRegularizationLayer(keras.layers.Layer):

    def __init__(self,rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


class OuterLayer(keras.layers.Layer):

  def __init__(self):
    super(OuterLayer, self).__init__()
    self.activity_reg = ActivityRegularizationLayer(1e-2)

  def call(self, inputs):
    return self.activity_reg(inputs)


layer = OuterLayer()
# Retrieve losses
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above


# layer.losses contiene le regularization loss di ogni layer interno
class OuterLayer2(keras.layers.Layer):

    def __init__(self):
        super(OuterLayer2, self).__init__()
        self.dense = keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayer2()
_ = layer(tf.zeros((1, 1)))

# ! This is `1e-3 * sum(layer.dense.kernel)`,
# created by the `kernel_regularizer` above.
print(layer.losses)

# ----------------------------------
# ---------- SO FAR SO ON ----------
# ----------------------------------

# Altri tutorial
# - Training and evaluation
# - https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/guide/keras/training_and_evaluation.ipynb
