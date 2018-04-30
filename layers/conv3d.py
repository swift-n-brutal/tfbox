import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Conv3d(Layer):
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=0, pad_type='CONSTANT',
            name='conv3d', filler=('msra', 0., 1.)):
        super(Conv3d, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            kd, kh, kw, kin, kout = filter_shape
            if pad_size == -1:
                pad_size = (kd - 1)/2
                pad_size_t = kd-1-pad_size
                x = tf.pad(x, [[0,0], [pad_size, pad_size_t], [pad_size, pad_size_t],
                    [pad_size, pad_size_t], [0, 0]], pad_type)
            elif pad_size > 0:
                x = tf.pad(x, [[0,0], [pad_size, pad_size], [pad_size, pad_size],
                    [pad_size, pad_size], [0,0]], pad_type)

            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                fan_in = kd * kh * kw * kin
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))

            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv3d(x, weight, [1, stride, stride, stride, 1], padding='VALID')
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

