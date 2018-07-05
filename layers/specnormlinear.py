import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layerupdateops import LayerUpdateOps
from .specnormconv2d import spec_norm_weight

#def _l2normalize(v, eps=1e-12):
#    return v / (tf.norm(v, ord=2) + eps)

#def spec_norm_weight(w, u, niter=1, stop_grad_sigma=True):
#    #
#    in_dim, out_dim = w.shape.as_list()
#    #
#    for i in xrange(niter):
#        v = _l2normalize(tf.matmul(u, tf.transpose(w)))
#        vw = tf.matmul(v, w) # v * w
#        u = _l2normalize(vw)
#    sigma = tf.matmul(vw, tf.transpose(u))[0,0]
#    if stop_grad_sigma:
#        sigma = tf.stop_gradient(sigma)
#    w_normalized = w / sigma
#    return w_normalized, u, sigma

class SpecNormLinear(LayerUpdateOps):
    """Spectral Normalized Linear (Fully Connected) Layer

    """
    def __init__(self, x, output_dim, bias=True,
            is_training=True, niter=1, stop_grad_sigma=False,
            name='sn_fc', filler=('msra', 0., 1.), update_collection=None):
        """__init__ method of SpecNormLinear

        Parameters
        ----------
        x : tf.Tensor
            Input of 4D tensor.
        output_dim : int
            Output dimension
        bias : bool
            Whether to add bias.
        is_training : bool
            If True, compute the max singular value using the power iteration method.
            Otherwise, use the stored value. 
        niter : int
            Number of iterations to compute the max singular value.
        stop_grad_sigma : bool
            Whether to treat sigma (the max singular value) as a constant.
        name : string
        filler : tuple
            Initializer for convolutional weight. One of
            -   ('msra', negative_slope, positive_slope)
            -   ('gaussian', mean, stdev)
            -   ('uniform', minval, maxval)
        update_collection : str or None
        """
        super(SpecNormLinear, self).__init__(name, update_collection)
        # inputs
        self.inputs.append(x)
        with tf.variable_scope(name) as scope:
            in_shape = x.shape.as_list()
            fan_in = np.prod(in_shape[1:])
            x = tf.reshape(x, [-1, fan_in])
            # initializer for weight
            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))
            # params
            weight = tf.get_variable('weight', shape=[fan_in, output_dim], dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            # update_params
            u = tf.get_variable('u', [1, output_dim], dtype=TF_DTYPE,
                    initializer=tf.truncated_normal_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', [], dtype=TF_DTYPE,
                    initializer=tf.constant_initializer(1.), trainable=False)
            self.update_params.extend([u, sigma])
            # normalize weight
            if is_training:
                weight_normalized, u_new, sigma_new = spec_norm_weight(weight, u, niter, stop_grad_sigma)
            else:
                weight_normalized = weight / sigma
                u_new, sigma_new = None, None
            # update_ops
            def get_update_ops(update_collection=update_collection):
                if self._update_ops is None:
                    self._update_ops = list()
                    with tf.name_scope(scope.original_name_scope):
                        with tf.name_scope(update_collection, default_name='default'):
                            self._update_ops.extend([u.assign(u_new), sigma.assign(sigma_new)])
                return self._update_ops
            if is_training:
                self.update_ops_getter = get_update_ops
            # linear
            y = tf.matmul(x, weight_normalized)

            if bias:
                b = tf.get_variable('bias', shape=[output_dim], dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                y = tf.nn.bias_add(y, b)
            # outputs
            self.outputs.append(y)
        self.print_info(LAYERS_VERBOSE)

