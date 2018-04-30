import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layerupdateops import LayerUpdateOps

class CondBatchNormBase(LayerUpdateOps):
    def __init__(self, x, y, eps=1e-5, moving_fraction=0.9,
            external_stats=None, output_stats=False,
            keep_moving_average=True, use_moving_average=False,
            name='cond_bnbase', update_collection=None):
        super(CondBatchNormBase, self).__init__(name, update_collection)
        #
        input_shape = x.shape.as_list() # n x h x w x chn
        cond_shape = y.shape.as_list() # n x cat
        param_shape = [cond_shape[-1], input_shape[-1]] # cat x chn
        affine_shape = [input_shape[0]] + [1] * (len(input_shape) - 2) + [input_shape[-1]]
        update_param_shape = input_shape[-1:]
        reduce_axes = range(len(input_shape) - 1)
        # inputs
        self.inputs.extend([x])
        if external_stats is not None:
            self.inputs.extend(external_stats)
        #
        with tf.variable_scope(name) as scope:
            # parameters for affine transformation conditioned on y
            # generally gamma should be positive and around 1, so we use exp to transform it
            beta_mat = tf.get_variable('beta_mat', param_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            beta_bias = tf.get_variable('beta_bias', param_shape[-1:], dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            log_gamma_mat = tf.get_variable('log_gamma_mat', param_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            log_gamma_bias = tf.get_variable('log_gamma_bias', param_shape[-1:], dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            self.params.extend([beta_mat, beta_bias, log_gamma_mat, log_gamma_bias])
            betas = tf.reshape(
                    tf.nn.bias_add(tf.matmul(y, beta_mat), beta_bias, name='betas'),
                    affine_shape)
            gammas = tf.reshape(
                    tf.exp(tf.nn.bias_add(tf.matmul(y, log_gamma_mat), log_gamma_bias), name='gammas'),
                    affine_shape)
            # update params
            def get_update_params(update_collection=update_collection):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    with tf.variable_scope(update_collection, default_name='default'):
                        _update_params = [
                                tf.get_variable('moving_mean', update_param_shape, dtype=TF_DTYPE,
                                    initializer=tf.constant_initializer(0.), trainable=False),
                                tf.get_variable('moving_variance', update_param_shape, dtype=TF_DTYPE,
                                    initializer=tf.constant_initializer(1.), trainable=False),
                                tf.get_variable('mult_moving_mean', [], dtype=TF_DTYPE,
                                    initializer=tf.constant_initializer(1.), trainable=False),
                                tf.get_variable('mult_moving_variance', [], dtype=TF_DTYPE,
                                    initializer=tf.constant_initializer(1.), trainable=False)]
                return _update_params
            if keep_moving_average or use_moving_average:
                self.update_params.extend(get_update_params())
            # get the current stats
            if external_stats is not None:
                assert len(external_stats) == 2, "external stats should contain two tensors (%d given)" % len(external_stats)
                cur_mean, cur_variance = external_stats
            elif keep_moving_average or not use_moving_average:
                n_var = np.prod(input_shape[:-1])
                var_factor = NP_DTYPE(1) if n_var == 1 else NP_DTYPE(n_var * 1. / (n_var - 1))
                cur_mean, cur_variance = tf.nn.moments(x, reduce_axes)
                cur_variance = var_factor * cur_variance
            else:
                # use moving average stats, do not update moving average stats, no external stats
                cur_mean, cur_variance = None, None
            # output current stats
            if output_stats:
                self.outputs.extend([cur_mean, cur_variance])
            # update ops
            def get_update_ops(update_collection=update_collection):
                if self._update_ops is None:
                    _update_params = self.update_params
                    self._update_ops = list()
                    with tf.name_scope(scope.original_name_scope):
                        with tf.name_scope(update_collection, default_name='default'):
                            self._update_ops.extend([
                                _update_params[0].assign(_update_params[0]*moving_fraction + cur_mean),
                                _update_params[1].assign(_update_params[1]*moving_fraction + cur_variance),
                                _update_params[2].assign(_update_params[2]*moving_fraction + 1),
                                _update_params[3].assign(_update_params[3]*moving_fraction + 1)])
                return self._update_ops
            if keep_moving_average:
                self.update_ops_getter = get_update_ops
            # compute the stats used by nn.batch_normalization
            if use_moving_average:
                update_params = self.update_params
                if keep_moving_average:
                    mean = (update_params[0]*moving_fraction + cur_mean) / (update_params[2]*moving_fraction + 1.)
                    variance = (update_params[1]*moving_fraction + cur_variance) / (update_params[3]*moving_fraction + 1.)
                else:
                    mean = update_params[0] / update_params[2]
                    variance = update_params[1] / update_params[3]
            else:
                mean, variance = cur_mean, cur_variance
            # batch normalization
            y = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            y = tf.add(tf.multiply(y, betas), gammas)
            # outputs
            self.outputs.append(y)
        self.print_info(LAYERS_VERBOSE)
            
# Original Batch Normalization
class CondBatchNorm(CondBatchNormBase):
    def __init__(self, x, y, phase, eps=1e-5, moving_fraction=0.9,
            name='cond_bn', update_collection=None):
        if phase == 'train':
            super(CondBatchNorm, self).__init__(x, y, eps=eps, moving_fraction=moving_fraction,
                    external_stats=None, output_stats=False,
                    keep_moving_average=True, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'train_no_update':
            super(CondBatchNorm, self).__init__(x, y, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'test':
            super(CondBatchNorm, self).__init__(x, y, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=True,
                    name=name, update_collection=update_collection)
        else:
            raise ValueError('Invalid phase: %s' % str(phase))

# Batch Normalization using external stats
class ExtCondBatchNorm(CondBatchNormBase):
    def __init__(self, x, y, phase, external_stats=None, eps=1e-5, moving_fraction=0.9,
            name='ext_cond_bn', update_collection=None):
        if phase == 'train':
            assert external_stats is not None, 'external stats should be provided in train phase'
            super(ExtCondBatchNorm, self).__init__(x, y, eps=eps, moving_fraction=moving_fraction,
                    external_stats=external_stats, output_stats=False,
                    keep_moving_average=True, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'stats':
            super(ExtCondBatchNorm, self).__init__(x, y, eps=eps, moving_fraction=moving_fraction,
                    external_stats=external_stats, output_stats=True,
                    keep_moving_average=False, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'test':
            super(ExtCondBatchNorm, self).__init__(x, y, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=True,
                    name=name, update_collection=update_collection)
        else:
            raise ValueError('INvalid phase: %s' % str(phase))
