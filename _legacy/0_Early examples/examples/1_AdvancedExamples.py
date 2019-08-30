import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://keras.io/getting-started/functional-api-guide/
# Keras API for advanced models

''' Simple, fully connected network (come prima, ma con l'API)'''
# 1 - Configurazione layer
inputs = keras.Input(shape=(32,))
#                            input del layer --|
#                                              v
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# 2 - Set up training
model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# /// random data
def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels
# ////


data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

# 3 - Actual training
model.fit(data, labels, batch_size=32, epochs=5)
