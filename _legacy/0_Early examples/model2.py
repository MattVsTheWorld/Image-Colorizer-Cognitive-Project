import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import os
import skimage.color as color
import skimage.io as io
import matplotlib.pyplot as plt
from Legacy.Old.preprocessing import ImageLoader


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


# function to take the selected batch of given size from input
def get_batch(elements, size, index):
    return elements[slice(index, index + size)]

def main():

    mydir = os.pardir + '/test_images/bird'
    images = [files for files in os.listdir(mydir)]
    image_size = 64
    img_format = 'jpeg'
    N = len(images)  # number of samples
    batch_size = 32  # number of images per batch
    data = np.zeros([N, image_size, image_size, 3])
    # resize to image_size x image_size
    '''
    for count in range(N):
        img = cv2.resize(io.imread(mydir + '/' + images[count]), (image_size, image_size))
        if count == 2:
            io.imshow(img)
            plt.show()
        data[count, :, :, :] = img
    '''
    num_train = N
    loader = ImageLoader(image_size, img_format)
    _, data = loader.load_folder(os.pardir + '/test_images/bird', mp=True)
    # normalization
    Xtrain = data[:num_train] * 1.0 / (image_size - 1)
    xt = Xtrain[:, :, :, 0]
    yt = Xtrain[:, :, :, 1:]
    yt = yt / image_size / 2
    xt = xt.reshape(num_train, image_size, image_size, 1)
    yt = yt.reshape(num_train, image_size, image_size, 2)

    session = tf.compat.v1.Session()

    # placeholders for samples and ground truth values
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, 1], name='x')
    ytrue = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, 2], name='ytrue')

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

    # loss function
    loss = tf.compat.v1.losses.mean_squared_error(labels=ytrue, predictions=conv13)
    # optimization metric
    cost = tf.reduce_mean(loss)
    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    session.run(tf.compat.v1.global_variables_initializer())

    num_epochs = 25

    # training
    total_batch = int(N / batch_size)
    for i in range(num_epochs):
        session.run(optimizer, feed_dict={x: xt, ytrue: yt})
        avg_loss = 0
        for j in range(total_batch):
            batch_x = get_batch(xt, batch_size, j)
            batch_y = get_batch(yt, batch_size, j)
            loss_value = session.run(cost, feed_dict={x: xt, ytrue: yt})
            avg_loss += loss_value / total_batch
        print("epoch: " + str(i) + " loss: " + str(avg_loss))

    output = session.run(conv13, feed_dict={x: xt[0].reshape([1, image_size, image_size, 1])}) * (image_size/2)
    image = np.zeros([image_size, image_size, 3])
    # reconstruct the image with the first column from L value and the second and third from the net output
    image[:, :, 0] = xt[2][:, :, 0]
    image[:, :, 1:] = output[0]
    image = color.lab2rgb(image)
    # io.imsave("test.jpg", image)
    print(image)
    io.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
