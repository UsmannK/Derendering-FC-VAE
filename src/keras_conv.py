from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = 256
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (int(image_size), int(image_size))
batch_size = 128
kernel_sizes = [8,8]
rectangularFilters = 4
squareFilters = 12
numberOfFilters = [10]
kernelSizes = [8,8]
poolSizes = [8,4]
poolStrides = [4,4]
filters = 16
latent_dim = 128
epochs = 30

# VAE model = encoder + decoder
# build encoder model
goal_image = Input(shape=input_shape, name='encoder_input_goal')
current_image = Input(shape=input_shape, name='encoder_input_current')
stacked_input = tf.stack([goal_image, current_image], axis=3)

horizontalKernels = Conv2D(
    filters = rectangularFilters,
    kernel_size= [kernelSizes[0]*2, int(kernelSizes[0]/2)],
    padding="same",
    activation='relu',
    strides = 1
)(stacked_input)

verticalKernels = Conv2D(
    filters = rectangularFilters,
    kernel_size= [int(kernelSizes[0]/2), kernelSizes[0]*2],
    padding="same",
    activation='relu',
    strides = 1
)(stacked_input)

squareKernels = Conv2D(
    filters = squareFilters,
    kernel_size = [kernelSizes[0]*2, int(kernelSizes[0]/2)],
    padding="same",
    activation='relu',
    strides = 1
)(stacked_input)

c1 = Concatenate(axis=3)([horizontalKernels, verticalKernels, squareKernels])
c1 = MaxPooling2D(pool_size = poolSizes[0], strides=poolStrides[0], padding='same')(c1)

numberOfFilters = numberOfFilters
kernelSizes = kernelSizes[1:]    
poolSizes = poolSizes[1:]
poolStrides = poolStrides[1:]
nextInput = c1

for filterCount,kernelSize,poolSize,poolStride in zip(numberOfFilters,kernelSizes,poolSizes,poolStrides):
    c1 = Conv2D(filters = filterCount,
                kernel_size = [kernelSize,kernelSize],
                padding = "same",
                activation = "relu",
                strides = 1)(nextInput)
    c1 = MaxPooling2D(pool_size = poolSize,
                    strides = poolStride,
                    padding = "same")(c1)
    nextInput = c1

flattened = Flatten()(c1)

x = Dense(1024, activation='relu')(flattened)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
# inputs = {'encoder_input_goal': [], 'encoder_input_current': []}
encoder = Model([goal_image, current_image], [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
d = Dense(128, activation='tanh')(latent_inputs)
d = Dropout(0.5)(d)
d = Dense(128, activation='tanh')(d)
d = Dropout(0.5)(d)
d = Dense(128, activation='relu')(d)
d = Dropout(0.5)(d)
d = Dense(128, activation='sigmoid')(d)
d = Dropout(0.5)(d)
d = Dense(128, activation='sigmoid')(d)
d = Dropout(0.5)(d)
d = Dense(128)(d)
d = Dropout(0.5)(d)
outputs = Dense(5, activation='sigmoid')(latent_inputs)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(stacked_input)[2])
vae = Model(inputs, outputs, name='vae')

# NB: the below are inner functions, not methods of Model
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    # eq 25
    z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
            2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = 1 - tf.square(rho)
    result = tf.exp(tf.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                    z_pen_logits, x1_data, x2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                            z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    result1 = -tf.log(result1 + epsilon)    # avoid log(0)

    fs = 1.0 - pen_data[:, 2]    # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)

    # result2: loss wrt pen state, (L_p in equation 9)
    result2 = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result2 = tf.reshape(result2, [-1, 1])
    if not self.hps.is_training:    # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

    result = result1 + result2
    return result

# below is where we need to do MDN (Mixture Density Network) splitting of
# distribution params
def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    z_pen_logits = z[:, 0:3]    # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
        z[:, 3:], 6, 1)

    # process output z's into MDN parameters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    r = [z_pi, z_mu1, z_mu2, z_sigma1,
            z_sigma2, z_corr, z_pen, z_pen_logits]
    return r

out = get_mixture_coef(output)
[o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

self.pi = o_pi
self.mu1 = o_mu1
self.mu2 = o_mu2
self.sigma1 = o_sigma1
self.sigma2 = o_sigma2
self.corr = o_corr
self.pen_logits = o_pen_logits
# pen state probabilities (result of applying softmax to self.pen_logits)
self.pen = o_pen

# reshape target data so that it is compatible with prediction shape
target = tf.reshape(self.output_x, [-1, 5])
tf.logging.info(target)
[x1_data, x2_data, eos_data, eoc_data,
    cont_data] = tf.split(target, 5, 1)
pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                        o_pen_logits, x1_data, x2_data, pen_data)

self.r_cost = tf.reduce_mean(lossfunc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    tf_sess = tf.Session()
    K.set_session(tf_sess)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_cnn_mnist.h5')

    plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")