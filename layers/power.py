import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Power(Layer):
    def __init__(self, x, y, name='power'):
        super(Power, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if y == 2:
                self.outputs.append(tf.square(x))
            else:
                self.outputs.append(tf.pow(x, y))
        self.print_info(LAYERS_VERBOSE)
            
