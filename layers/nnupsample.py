import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class NNUpsample(Layer):
    def __init__(self, x, factor, name='nnupsample'):
        super(NNUpsample, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            output_size = (shape[1] * factor, shape[2] * factor)
            self.outputs.append(tf.image.resize_nearest_neighbor(x, size=output_size))
        self.print_info(LAYERS_VERBOSE)

