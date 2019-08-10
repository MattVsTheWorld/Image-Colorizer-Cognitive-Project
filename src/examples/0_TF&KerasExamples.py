import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------------
# ----- Esempio 1 -----
# ---------------------
'''
model = keras.models.Sequential([keras.layers.Dense(1, input_dim=1)])

model.compile(loss='mean_squared_error', optimizer='sgd')
xs = np.array([-1, 0, 1, 2, 3, 4])
ys = np.array([-3, -1, 1, 3, 5, 7])
model.fit(xs, ys, epochs=500)

to_predict = np.array([10, 11, 12, 13])
print(model.predict(to_predict))
'''

# ---------------------
# ----- Esempio 2 -----
# ---------------------

# 1 - Configurazione layer
# Esempio di Fully-connected network sequenziale
# Si possono aggiungere poi (model.add), o inizializzarlo così
model = keras.Sequential([
    # Layer densely-connected con 64 unità
    keras.layers.Dense(64, activation='relu'),
    # or
    keras.layers.Dense(64, activation=tf.nn.relu),
    # Esempio di regolarizzazione l1 alla kernel matrix
    # https://stats.stackexchange.com/questions/383310/difference-between-kernel-bias-and-activity-regulizers-in-keras
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01)),
    # 10 unità di output, softmax
    keras.layers.Dense(10, activation='softmax')
])

# 2 - Set up training
model.compile(
            # Procedura di training. Adam è un'estensione del gradient descent stocastico.
              optimizer=tf.train.AdamOptimizer(0.001),
            # La funzione di loss da minimizzare durante l'ottimizzazione.
              loss='categorical_crossentropy',
            # metriche che analizzano il training
              metrics=['accuracy'])


# DATASET PICCOLI
# Mette i dati in array numpy
def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


# Genera un po' di dati...
data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))
# Validation data...
val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))


# 3.1 - Actual training
model.fit(data, labels,
          # 1 epoch = 1 iterazione su tutto il dataset (in piccoli batch)
          epochs=10,
          # (w/ numpy data) divide i dati in batch piu' piccoli
          # itera su questi durante il training. Se non divisibile correttamente, l'ultimo batch è il più piccolo
          batch_size=32,
          # Dati di validazione su cui monitorare la performance.
          # Tupla di input e labels
          validation_data=(val_data, val_labels)
          )

# DATASET GRANDI (?)
# Toy dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# 3.2 - Different training
model.fit(dataset, epochs=10,
          # numero di step di training che fa prima di cambiare epoch
          steps_per_epoch=30)

# 4 - Evaluate / predict
data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
print(result)

# Example INFO log
tf.compat.v1.logging.info('Log test')
