import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Concat(Layer):
    def __init__(self, xs, axis=-1, name='concat'):
        super(Concat, self).__init__(name)
        self.inputs.extend(xs)
        with tf.variable_scope(name):
            self.outputs.append(tf.concat(xs, axis))
        self.print_info(LAYERS_VERBOSE)

