import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Add(Layer):
    def __init__(self, xs, coeffs=None, name='add'):
        super(Add, self).__init__(name)
        self.inputs.extend(xs)
        with tf.variable_scope(name):
            if coeffs is not None:
                assert type(coeffs) is list, "Invalid coeffs: %s" % type(coeffs)
                assert len(coeffs) == len(xs), "Must have same length: (%d, %d)" % \
                        (len(coeffs), len(xs))
                for (i, c) in enumerate(coeffs):
                    xs[i] = xs[i] * c
            self.outputs.append(tf.add_n(xs))
        self.print_info(LAYERS_VERBOSE)
            
