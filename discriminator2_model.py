from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import
import os
import urllib
import numpy as np
import tarfile
from past.builtins import xrange
from tensorflow.python.platform import gfile
import tensorflow as tf

import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import pdb
import tensorflow as tf
import numpy as np

import functools

def apply_conv(x, filters, kernel_size=3, he_init=True):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                            padding='SAME', kernel_initializer=initializer)


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(x,
                                    decay=0.9,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-5,
                                    zero_debias_moving_mean=True,
                                    is_training=is_training)


def stable_norm(x, ord):
    x = tf.contrib.layers.flatten(x)
    alpha = tf.reduce_max(tf.abs(x) + 1e-5, axis=1)
    result = alpha * tf.norm(x / alpha[:, None], ord=ord, axis=1)
    return result


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n([x[:,::2,::2,:], x[:,1::2,::2,:],
                         x[:,::2,1::2,:], x[:,1::2,1::2,:]]) / 4.

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))

def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)

def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)

def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(conv_meanpool, filters=filters)
        conv_shortcut = functools.partial(conv_meanpool, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = functools.partial(upsample_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = functools.partial(upsample_conv, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1((norm_fn(x)))
        update = conv_2((norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool((update), filters=filters)

        skip = meanpool_conv(x, filters=32, kernel_size=1, he_init=False)
        return skip + update
#########################
def discriminator_2(x,y, reuse):
    with tf.variable_scope('discriminator2', reuse=reuse):
        with tf.name_scope('pre_process'):
            x2 = tf.layers.conv2d(inputs = x,filters = 32,kernel_size=3,padding="same")
            x23 = tf.layers.conv2d(inputs = x2,filters = 16,kernel_size=3,padding="same")
            x233 = tf.layers.conv2d(inputs = x23,filters = 8,kernel_size=3,padding="same")
            x2334 = tf.layers.conv2d(inputs = x233,filters = 1,kernel_size=3,padding="same")
        
            #print(x.shape)
            #pdb.set_trace()
        
        
        

        
            loss = tf.reduce_mean(tf.abs(x2334-y))
            #x7 = tf.reduce_mean(x6, axis=[1, 2])
            #flat2 = tf.contrib.layers.flatten(x7)
            #flat = tf.layers.dense(flat2, 1)
            return loss
