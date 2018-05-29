import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Act(Layer):
    def __init__(self, x, args, name=None):
        kwargs = dict()
        #
        if name is not None:
            kwargs['name'] = name
        #
        act_type = args[0]
        if args[0] == 'ReLU':
            if len(args) == 2:
                kwargs['slope'] = args[1]
            self.ReLU(x, **kwargs)
        elif args[0] == 'ELU':
            if len(args) == 2:
                kwargs['alpha'] = args[1]
            self.ELU(x, **kwargs)
        else:
            raise ValueError('Invalid activation type: %s' % str(args[0]))
        self.print_info(LAYERS_VERBOSE)

    def ELU(self, x, alpha=1., name='elu'):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            elu1 = tf.nn.elu(x)
            if alpha == 1.:
                self.outputs.append(elu1)
            else:
                self.outputs.append(tf.where(x >= 0, elu1, alpha * elu1))

    def ReLU(self, x, slope=0., name='relu'):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if slope == 0.:
                self.outputs.append(tf.nn.relu(x))
            else:
                self.outputs.append(tf.where(x >= 0, x, slope*x))

