import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layerupdateops import LayerUpdateOps

def _l2normalize(v, eps=1e-12):
    return v / (tf.norm(v, ord=2) + eps)

def spec_norm_weight(w, u, niter=1, stop_grad_sigma=True):
    #
    w_shape = w.shape.as_list()
    out_dim = w_shape[-1] # the last chn is treated as output dim
    in_dim = np.prod(w_shape[:-1])
    w_reshaped = tf.reshape(w, [-1, out_dim])
    #
    for i in xrange(niter):
        v = _l2normalize(tf.matmul(u, tf.transpose(w_reshaped)))
        vw = tf.matmul(v, w_reshaped) # v * w
        u = _l2normalize(vw)
    sigma = tf.matmul(vw, tf.transpose(u))[0,0]
    if stop_grad_sigma:
        sigma = tf.stop_gradient(sigma)
    w_normalized = w / sigma
    return w_normalized, u, sigma

class SpecNormConv2d(LayerUpdateOps):
    """Spectral Normalized 2D Convolutional Layer

    """
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=0, pad_mode='CONSTANT',
            is_training=True, niter=1, stop_grad_sigma=True,
            name='sn_conv2d', filler=('msra', 0., 1.), update_collection=None):
        """__init__ method of SpecNormConv2d

        Parameters
        ----------
        x : tf.Tensor
            Input of 4D tensor.
        filter_shape : list of int
            Shape of convolutional filter. [filter_height, filter_width, chn_in]
            or [filter_height, filter_width, chn_in, chn_out]
        bias : bool
            Whether to add bias.
        stride : int
            Stride size.
        pad_size : int
        pad_mode : str
            One of 'CONSTANT', 'REFLECT', or 'SYMMETRIC'
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
        super(SpecNormConv2d, self).__init__(name, update_collection)
        # inputs
        self.inputs.append(x)
        in_shape = x.shape.as_list()
        if len(filter_shape) == 3:
            # get chn_in from input tensor
            kin = in_shape[-1]
            kout = filter_shape[-1]
            filter_shape[-1] = kin
            filter_shape.append(kout)
        kh, kw, kin, kout = filter_shape
        with tf.variable_scope(name) as scope:
            # padding
            padding = 'VALID'
            if pad_size == -1:
                # 'SAME' padding
                if pad_mode == 'CONSTANT':
                    padding = 'SAME'
                else:
                    w_in = in_shape[-2]
                    if w_in % stride == 0:
                        pad_size_both = max(kw - stride, 0)
                    else:
                        pad_size_both = max(kw - (w_in % stride), 0)
                    if pad_size_both > 0:
                        pad_size = pad_size_both / 2
                        x = tf.pad(x, [[0,0], [pad_size, pad_size_both-pad_size],
                            [pad_size, pad_size_both-pad_size], [0,0]], pad_mode)
            elif pad_size > 0:
                # pad_size padding on both sides of each dimension
                x = tf.pad(x, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], pad_mode)
            # initializer for convolutional kernel
            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                fan_in = kh * kw * kin
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))
            # params
            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            # update_params
            u = tf.get_variable('u', [1, kout], dtype=TF_DTYPE,
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
            # udpate_ops
            def get_update_ops(update_collection=update_collection):
                if self._update_ops is None:
                    self._update_ops = list()
                    with tf.name_scope(scope.original_name_scope):
                        with tf.name_scope(update_collection, default_name='default'):
                            self._update_ops.extend([u.assign(u_new), sigma.assign(sigma_new)])
                return self._update_ops
            if is_training:
                self.update_ops_getter = get_update_ops
            # conv2d
            y = tf.nn.conv2d(x, weight_normalized, [1, stride, stride, 1], padding=padding)
            # add channel-wise bias
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                y = tf.nn.bias_add(y, b)
            # outputs
            self.outputs.append(y)
        self.print_info(LAYERS_VERBOSE)
