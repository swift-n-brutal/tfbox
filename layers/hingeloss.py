import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class HingeLoss(Layer):
    def __init__(self, y, t, name='hinge_loss'):
        super(HingeLoss, self).__init__(name)
        self.inputs.extend([y])
        if t != 1 and t != -1:
            raise ValueError('Invalid label: t = {}'.format(t))
        with tf.variable_scope(name):
            if t == 1:
                loss = tf.reduce_mean(tf.maximum(1 - y, 0))
            else:
                loss = tf.reduce_mean(tf.maximum(1 + y, 0))
            self.outputs.append(loss)
        self.print_info(LAYERS_VERBOSE)

