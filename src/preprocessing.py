from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

# Allows INFO logs to be printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Set verbosity of warnings allowed (higher number, less warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img = cv2.cvtColor(cv2.imread(os.pardir + '/imagenet/a_subfolder/twitch.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)

imgBW = img.copy()
for i in range(0, 256):
    for j in range(0, 256):
        imgBW[i][j] = imgBW[i][j].take(1)
        
cv2.imshow('img', imgBW)
cv2.waitKey()
