import tensorflow as tf
import numpy as np

def make_model(imageInpt):
    inputSize = 256
    squareFilters = 12
    rectangularFilters = 4
    numberOfFilters = [10]
    kernelSizes = [8,8]
    poolSizes = [8,4]
    poolStrides = [4,4]
    if imageInpt.shape[1] != inputSize:
        imageInpt = tf.image.resize_bilinear(imageInpt, [inputSize]*2)
    shape = imageInpt.shape
    imageInpt = tf.reshape(imageInpt, (shape[0], shape[1], shape[2], 1))

    horizontalKernels = tf.layers.conv2d(inputs = imageInpt,
                                            filters = rectangularFilters,
                                            kernel_size = [kernelSizes[0]*2,
                                                        kernelSizes[0]/2],
                                            padding = "same",
                                            activation = tf.nn.relu,
                                            strides = 1)
    verticalKernels = tf.layers.conv2d(inputs = imageInpt,
                                            filters = rectangularFilters,
                                            kernel_size = [kernelSizes[0]/2,
                                                        kernelSizes[0]*2],
                                            padding = "same",
                                            activation = tf.nn.relu,
                                            strides = 1)
    squareKernels = tf.layers.conv2d(inputs = imageInpt,
                                            filters = squareFilters,
                                            kernel_size = [kernelSizes[0],
                                                        kernelSizes[0]],
                                            padding = "same",
                                            activation = tf.nn.relu,
                                            strides = 1)
    c1 = tf.concat([horizontalKernels,verticalKernels,squareKernels], axis = 3)
    c1 = tf.layers.max_pooling2d(inputs = c1,
                                    pool_size = poolSizes[0],
                                    strides = poolStrides[0],
                                    padding = "same")

    numberOfFilters = numberOfFilters
    kernelSizes = kernelSizes[1:]    
    poolSizes = poolSizes[1:]
    poolStrides = poolStrides[1:]
    nextInput = c1
    for filterCount,kernelSize,poolSize,poolStride in zip(numberOfFilters,kernelSizes,poolSizes,poolStrides):
        c1 = tf.layers.conv2d(inputs = nextInput,
                                filters = filterCount,
                                kernel_size = [kernelSize,kernelSize],
                                padding = "same",
                                activation = tf.nn.relu,
                                strides = 1)
        c1 = tf.layers.max_pooling2d(inputs = c1,
                                        pool_size = poolSize,
                                        strides = poolStride,
                                        padding = "same")
        nextInput = c1
    return nextInput

# def build_conv_model(self, hps):
#     if hps.is_training:
#         self.global_step = tf.Variable(
#             0, name='global_step', trainable=False)
#     cell_fn = rnn.LSTMCell

#     use_recurrent_dropout = True
#     use_input_dropout = False
#     use_output_dropout = False

#     cell = cell_fn(
#         hps.dec_rnn_size,
#         use_recurrent_dropout=use_recurrent_dropout,
#         # dropout_keep_prob=self.hps.recurrent_dropout_prob)
#         dropout_keep_prob=0.90)

#     self.sequence_lengths = tf.placeholder(
#         dtype=tf.int32, shape=[self.hps.batch_size])
    
#     self.current_batch = tf.placeholder(tf.float32, [self.hps.batch_size, 256, 256])
#     self.goal_batch = tf.placeholder(tf.float32, [self.hps.batch_size, 256, 256])

#     # self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]

#     if hps.conditional: #vae mode
#         self.mean, self.presig = self.cnn_encoder(self.goal_batch, self.current_batch)

#         # sigma > 0. div 2.0 -> sqrt.
#         self.sigma = tf.exp(self.presig / 2.0)
#         eps = tf.random_normal(
#             (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
#         self.batch_z = self.mean + tf.multiply(self.sigma, eps)
#         # KL cost
#         self.kl_cost = -0.5 * tf.reduce_mean(
#             (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
#         self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
#         pre_tile_y = tf.reshape(self.batch_z,
#                                 [self.hps.batch_size, 1, self.hps.z_size])

