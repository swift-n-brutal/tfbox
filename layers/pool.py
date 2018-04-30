import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Pool(Layer):
    def __init__(self, x, pool_type, ksize, stride=1, padding='VALID', name='pool'):
        super(Pool, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if pool_type == 'AVG':
                self.outputs.append(tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1], padding=padding))
            elif pool_type == 'MAX':
                self.outputs.append(tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1], padding=padding))
            else:
                raise ValueError('Invalid pool type: %s' % (pool_type))
        self.print_info(LAYERS_VERBOSE)

