import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Reshape(Layer):
    def __init__(self, x, new_shape, name='reshape'):
        super(Reshape, self).__init__(name)
        self.inputs.extend([x])
        with tf.variable_scope(name):
            self.outputs.append(tf.reshape(x, new_shape))
        self.print_info(LAYERS_VERBOSE)

