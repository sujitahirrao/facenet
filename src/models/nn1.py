from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# with reference to Maxout Network
def maxout(layer_input, n_maxouts=1):
    with tf.variable_scope('maxout'):
        layer_output = None
        for i in range(n_maxouts):
            fc = slim.fully_connected(layer_input, num_outputs=4096, activation_fn=None)
            if layer_output is None:
                layer_output = fc
            else:
                layer_output = tf.maximum(layer_output, fc)
        return layer_output


# maxout with dimension reduction
def maxout_reduction(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # return nn1(images)
        return nn1(images, is_training=phase_train, reuse=reuse)


# NN1 implementation
def nn1(inputs, is_training=True, reuse=None, scope='NN1'):
    with tf.variable_scope(scope, 'NN1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='maxpool1')
                net = slim.nn.lrn(net, name='lrn1')

                net = slim.conv2d(net, num_outputs=64, kernel_size=1, stride=1, scope='conv2a')
                net = slim.conv2d(net, num_outputs=192, kernel_size=3, stride=1, scope='conv2')
                net = slim.nn.lrn(net, name='lrn2')
                net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='maxpool2')

                net = slim.conv2d(net, num_outputs=192, kernel_size=1, stride=1, scope='conv3a')
                net = slim.conv2d(net, num_outputs=384, kernel_size=3, stride=1, scope='conv3')
                net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='maxpool3')

                net = slim.conv2d(net, num_outputs=384, kernel_size=1, stride=1, scope='conv4a')
                net = slim.conv2d(net, num_outputs=256, kernel_size=3, stride=1, scope='conv4')

                net = slim.conv2d(net, num_outputs=256, kernel_size=1, stride=1, scope='conv5a')
                net = slim.conv2d(net, num_outputs=256, kernel_size=3, stride=1, scope='conv5')

                net = slim.conv2d(net, num_outputs=256, kernel_size=1, stride=1, scope='conv6a')
                net = slim.conv2d(net, num_outputs=256, kernel_size=3, stride=1, scope='conv6')
                net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='maxpool4')

                # concat layer
                net = slim.flatten(net)

                # fc1 with maxout pooling size as p = 2
                with tf.variable_scope('fc1'):
                    net = maxout(net, n_maxouts=2)

                # fc2 with maxout pooling size as p = 2
                with tf.variable_scope('fc2'):
                    net = maxout(net, n_maxouts=2)

                '''
                net = slim.fully_connected(net, num_outputs=4096, activation_fn=None, scope='fc1')
                net = maxout_reduction(net, 4096)

                net = slim.fully_connected(net, num_outputs=4096, activation_fn=None, scope='fc2')
                net = maxout_reduction(net, 4096)
                '''
                
                net = slim.fully_connected(net, num_outputs=128, activation_fn=None, scope='fc7128')

                # L2 Layer is as follows. In this repos it is written below calling function
                # embeddings = tf.nn.l2_normalize(net, 1, 1e-10, name='embeddings')

    print("\n\nNN1 built!\n\n")

    return net, None
