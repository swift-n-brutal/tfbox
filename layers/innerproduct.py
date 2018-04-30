import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class InnerProduct(Layer):
    def __init__(self, x, y, keepdims=True, name='inner_prod'):
        super(InnerProduct, self).__init__(name)
        self.inputs.append(x)
        x_shape = x.shape.as_list()
        y_shape = y.shape.as_list()
        assert x_shape == y_shape, 'Inconsistent shape: {} vs {}'.format(x_shape, y_shape)
        reduce_axis = range(1, len(x_shape))
        with tf.variable_scope(name):
            self.outputs.append(tf.reduce_sum(x*y, axis=reduce_axis, keepdims=keepdims))
        self.print_info(LAYERS_VERBOSE)

