import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Act(Layer):
    def __init__(self, x, args, name=None):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        if args[0] == 'ReLU':
            if len(args) == 1:
                self.ReLU(x, name=name)
            elif len(args) == 2:
                self.ReLU(x, args[1], name=name)
        elif args[0] == 'ELU':
            if len(args) == 1:
                self.ELU(x, name=name)
            elif len(args) == 2:
                self.ELU(x, args[1], name=name)
        else:
            raise ValueError('Invalid activation type: %s' % str(args[0]))
        self.print_info(LAYERS_VERBOSE)

    def ELU(self, x, alpha=1., name='elu'):
        with tf.variable_scope(name, default_name='elu'):
            elu1 = tf.nn.elu(x)
            if alpha == 1.:
                self.outputs.append(elu1)
            else:
                self.outputs.append(tf.where(x >= 0, elu1, alpha * elu1))

    def ReLU(self, x, slope=0., name='relu'):
        with tf.variable_scope(name, default_name='relu'):
            if slope == 0.:
                self.outputs.append(tf.nn.relu(x))
            else:
                self.outputs.append(tf.where(x >= 0, x, slope*x))

