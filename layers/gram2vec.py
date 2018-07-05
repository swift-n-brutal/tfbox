import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

def _l2normalize(vs, eps=1e-12):
    return vs / (tf.norm(vs, axis=0, ord=2) + eps)

class Gram2Vec(Layer):
    def __init__(self, gram_mat, output_dim, normalized=False, name='gram2vec'):
        super(Gram2Vec, self).__init__(name)
        self.inputs.append(gram_mat)
        shape = gram_mat.shape.as_list()
        assert len(shape) == 3 and shape[1] == shape[2], 'Input shape should be [n, c, c] ({} given)'.format(shape)
        batch_size = shape[0]
        chn = shape[1]
        with tf.variable_scope(name):
            # params
            stdev = np.sqrt(1. / chn) if normalized else 0.02
            initializer = tf.random_normal_initializer(0., stdev)
            u = tf.get_variable('u', shape=(chn, output_dim), dtype=TF_DTYPE, initializer=initializer)
            self.params.append(u)
            #
            if normalized:
                u = _l2normalize(u)
            # v = ( ... <u_i, Gu_i> ... )
            gram_mat = tf.reshape(gram_mat, [-1, chn]) # shape = [n, c, c] -> [n*c, c]
            Gu = tf.matmul(gram_mat, u, name='Gu') # shape = [n*c, d]
            Gu = tf.reshape(Gu, [batch_size, chn, output_dim]) # shape = [n, c, d]
            v = tf.reduce_sum(tf.multiply(u, Gu), axis=1, name='v') # shape = [n, d]
            self.outputs.append(v)
        self.print_info(LAYERS_VERBOSE)

