import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 0 - Creiamo un modello
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 1 - Salva solo i pesi
# Salva i pesi in un TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Per ricaricare il modello (chiaramente, il modello deve avere la stessa architettura)
model.load_weights('./weights/my_model')

# Si possono salvare anche in formato Keras HDF5
# model.save_weights('my_model.h5', save_format='h5')
# model.load_weights('my_model.h5')

# 2 - Salva solo la configurazione (senza pesi)
# Per ricreare lo stesso modello anche senza C O D E

# Serializza
json_string = model.to_json()

import json
import pprint
pprint.pprint(json.loads(json_string))

# Ricrea
fresh_model = keras.models.model_from_json(json_string)
# Si può fare anche con YAML...


# 3 - Salva l'intero modello
def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

model2 = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    keras.layers.Dense(10, activation='softmax')
])
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
model2.fit(data, labels, batch_size=32, epochs=5)

# salva tutto in HDF5
model2.save('./weights/my_model.h5')

# ricrea l'intero modello, includendo pesi e ottimizzatore
model2 = keras.models.load_model('./weights/my_model.h5')
