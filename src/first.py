import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1, input_dim=1))

model.compile(loss='mean_squared_error', optimizer='sgd')
xs = np.array([-1, 0, 1, 2, 3, 4])
ys = np.array([-3, -1, 1, 3, 5, 7])
model.fit(xs, ys, epochs=500)

to_predict = np.array([10, 11, 12, 13])
print(model.predict(to_predict))
