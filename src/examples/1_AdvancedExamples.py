import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'