import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# -----------------------------------------------------------------------------
# -------------------------------- DEFINIZIONI --------------------------------
# -----------------------------------------------------------------------------
# 1 - Model subclassing
# Per creare un modello personalizzato per il forward pass.
# Utile per la eager execution, in quanto si puo' scrivere il pass in imperative-style
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

    # build crea i pesi del layer. Si aggiungono con add_weight
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # crea variabile peso trainabile
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # NECESSARIO: chiamare il super metodo build
        super(MyLayer, self).build(input_shape)

    # definisce il forward pass
    def call(self, inputs):
        # prodotto matriciale
        return tf.matmul(inputs, self.kernel)

    # Specifica come computare l'output shape del layer, dato l'input shape (di un tensore)
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    # (Opzionale) serializza il layer (per modelli funzionali) (see below)
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 3 - Callbacks
# Oggetti passati a modelli per avere dei comportamenti custom durante il training
# Si possono fare custom, oppure usare i suoi:
# ModelCheckpoint       - salva checkpoint del modello a intervalli regolari
# LearningRateScheduler - cambia dinamicamente il learning rate
# EarlyStopping         - interrompe training quando la performance validation smette di migliorare
# TensorBoard           - per monitorare il modello con TensorBoard

callbacks = [
    # Interrompe training se val_loss non migliora dopo 2 epochs
    keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Scrive log di tensorboard a './logs'
    keras.callbacks.TensorBoard(log_dir='./logs')
]
"""                                                            ---|
#                                                               v
# model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
#           validation_data=(val_data, val_labels))
"""
# --------------------------------------------------------------------------
# -------------------------------- UTILIZZO --------------------------------
# --------------------------------------------------------------------------
# 1 - Utilizzare il modello
# Instance the model
model = MyModel(num_classes=15)

# 2.1 - Esempio di get_config
layer = MyLayer(keras.layers.Layer)
config = layer.get_config()
print(config)
# Ora si può ricreare dalla configurazione
new_layer = MyLayer.from_config(config)

# 2.2 - Utilizzare un modello con custom layers
model2 = keras.Sequential([MyLayer(10), keras.layers.Activation('softmax')])

model2.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# model.fit(data, labels, batch_size=32, epochs=5)
