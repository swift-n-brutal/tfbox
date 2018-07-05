import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

def _l2normalize(vs, eps=1e-12):
    return vs / (tf.norm(vs, axis=0, ord=2) + eps)

class Vec2Gram(Layer):
    def __init__(self, v, output_chn, normalized=False, name='gram2vec'):
        super(Vec2Gram, self).__init__(name)
        self.inputs.append(v)
        shape = v.shape.as_list()
        assert len(shape) == 2, 'Input shape should be [n, d] ({} given)'.format(shape)
        batch_size = shape[0]
        input_dim = shape[1]
        with tf.variable_scope(name):
            # params
            stdev = np.sqrt(1. / output_chn) if normalized else 0.02
            initializer = tf.random_normal_initializer(0., stdev)
            u = tf.get_variable('u', shape=(output_chn, input_dim), dtype=TF_DTYPE, initializer=initializer)
            self.params.append(u)
            #
            if normalized:
                u = _l2normalize(u)
            # G = \sum_i v_i\cdot u_i u_i^T
            v = tf.expand_dims(v, 1) # shape = [n, d] -> [n, 1, d]
            uv = tf.multiply(u, v, name='uv') # shape = [n, c, d]
            uv = tf.reshape(uv, [-1, input_dim]) # shape = [n*c, d]
            gram_mat = tf.matmul(uv, u, transpose_b=True, name='G') # shape = [n*c, c]
            gram_mat = tf.reshape(gram_mat, [batch_size, output_chn, output_chn]) # shape = [n, c, c]
            self.outputs.append(gram_mat)
        self.print_info(LAYERS_VERBOSE)

