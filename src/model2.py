import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import skimage.color as color
import skimage.io as io


# function to obtain the set of weights from a shape
def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))


# function to create the tensor of the layer's bias
def create_bias(size):
    return tf.Variable(tf.constant(0.1, shape=[size]))


def convolution(inputs, num_channels, filter_size, num_filters):
    # create weights
    weights = create_weights(shape=[filter_size, filter_size, num_channels, num_filters])
    bias = create_bias(num_filters)  # apply bias

    # convolutional layer
    layer = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + bias
    layer = tf.nn.tanh(layer)  # activation function
    return layer

# pooling function
def maxpool(inputs, kernel, stride):
    layer = tf.nn.max_pool(value=inputs, ksize=[1, kernel, kernel, 1],
                           strides=[1, stride, stride, 1], padding="SAME")
    return layer


# upsampling function
def upsampling(inputs):
    layer = tf.compat.v1.image.resize_nearest_neighbor(inputs,
                                             (2 * inputs.get_shape().as_list()[1], 2 * inputs.get_shape().as_list()[2]))
    return layer


def main():

    mydir = os.pardir + '/test_imgs/bird'
    images = [files for files in os.listdir(mydir)]
    N = len(images)  # number of samples
    data = np.zeros([N, 256, 256, 3])
    # resize to 256 x 256
    for count in range(N):
        img = cv2.resize(io.imread(mydir + '/' + images[count]), (256, 256))
        data[count, :, :, :] = img
    num_train = N

    # normalization
    Xtrain = color.rgb2lab(data[:num_train] * 1.0 / 255)
    xt = Xtrain[:, :, :, 0]
    yt = Xtrain[:, :, :, 1:]
    yt = yt / 128
    xt = xt.reshape(num_train, 256, 256, 1)
    yt = yt.reshape(num_train, 256, 256, 2)

    session = tf.compat.v1.Session()

    # placeholders for samples and ground truth values
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 256, 256, 1], name='x')
    ytrue = tf.compat.v1.placeholder(tf.float32, shape=[None, 256, 256, 2], name='ytrue')

    # layers
    conv1 = convolution(x, 1, 3, 3)
    max1 = maxpool(conv1, 2, 2)
    conv2 = convolution(max1, 3, 3, 8)
    max2 = maxpool(conv2, 2, 2)
    conv3 = convolution(max2, 8, 3, 16)
    max3 = maxpool(conv3, 2, 2)
    conv4 = convolution(max3, 16, 3, 16)
    max4 = maxpool(conv4, 2, 2)
    conv5 = convolution(max4, 16, 3, 32)
    max5 = maxpool(conv5, 2, 2)
    conv6 = convolution(max5, 32, 3, 32)
    max6 = maxpool(conv6, 2, 2)
    conv7 = convolution(max6, 32, 3, 64)
    upsample1 = upsampling(conv7)
    conv8 = convolution(upsample1, 64, 3, 32)
    upsample2 = upsampling(conv8)
    conv9 = convolution(upsample2, 32, 3, 32)
    upsample3 = upsampling(conv9)
    conv10 = convolution(upsample3, 32, 3, 16)
    upsample4 = upsampling(conv10)
    conv11 = convolution(upsample4, 16, 3, 16)
    upsample5 = upsampling(conv11)
    conv12 = convolution(upsample5, 16, 3, 8)
    upsample6 = upsampling(conv12)
    conv13 = convolution(upsample6, 8, 3, 2)

    # loss funtion
    loss = tf.compat.v1.losses.mean_squared_error(labels=ytrue, predictions=conv13)
    # optimization metric
    cost = tf.reduce_mean(loss)
    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    session.run(tf.global_variables_initializer())

    num_epochs = 100

    # training
    for i in range(num_epochs):
        session.run(optimizer, feed_dict={x: xt, ytrue: yt})
        lossvalue = session.run(cost, feed_dict={x: xt, ytrue: yt})
        print("epoch: " + str(i) + " loss: " + str(lossvalue))

    output = session.run(conv13, feed_dict={x: xt[0].reshape([1, 256, 256, 1])}) * 128
    image = np.zeros([256, 256, 3])
    image[:, :, 0] = xt[0][:, :, 0]
    image[:, :, 1:] = output[0]
    image = color.lab2rgb(image)
    io.imsave("test.jpg", image)
    cv2.imshow('lul', image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
