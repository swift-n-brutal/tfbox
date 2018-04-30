import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class SumLoss(Layer):
    def __init__(self, x, mean=True, name='sum_loss'):
        super(SumLoss, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if mean:
                self.outputs.append(tf.reduce_mean(x))
            else:
                self.outputs.append(tf.reduce_sum(x))
        self.print_info(LAYERS_VERBOSE)

