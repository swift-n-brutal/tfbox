import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layerupdateops import LayerUpdateOps

class BatchNormBase(LayerUpdateOps):
    def __init__(self, x, eps=1e-5, moving_fraction=0.9,
            external_stats=None, output_stats=False,
            keep_moving_average=True, use_moving_average=False,
            name='bnbase', update_collection=None):
        super(BatchNormBase, self).__init__(name, update_collection)
        #
        input_shape = x.get_shape().as_list()
        param_shape = input_shape[-1:]
        update_param_shape = input_shape[-1:]
        reduce_axes = range(len(input_shape) - 1)
        # inputs
        self.inputs.append(x)
        if external_stats is not None:
            self.inputs.extend(external_stats)
        #
        with tf.variable_scope(name) as scope:
            # parameters for affine transformation
            beta = tf.get_variable('beta', param_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable('gamma', param_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(1.))
            self.params.extend([beta, gamma])
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
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            # outputs
            self.outputs.append(y)
        self.print_info(LAYERS_VERBOSE)
            
# Original Batch Normalization
class BatchNorm(BatchNormBase):
    def __init__(self, x, phase, eps=1e-5, moving_fraction=0.9,
            name='bn', update_collection=None):
        if phase == 'train':
            super(BatchNorm, self).__init__(x, eps=eps, moving_fraction=moving_fraction,
                    external_stats=None, output_stats=False,
                    keep_moving_average=True, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'train_no_update':
            super(BatchNorm, self).__init__(x, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'test':
            super(BatchNorm, self).__init__(x, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=True,
                    name=name, update_collection=update_collection)
        else:
            raise ValueError('Invalid phase: %s' % str(phase))

# Batch Normalization using external stats
class ExtBatchNorm(BatchNormBase):
    def __init__(self, x, phase, external_stats=None, eps=1e-5, moving_fraction=0.9,
            name='ext_bn', update_collection=None):
        if phase == 'train':
            assert external_stats is not None, 'external stats should be provided in train phase'
            super(ExtBatchNorm, self).__init__(x, eps=eps, moving_fraction=moving_fraction,
                    external_stats=external_stats, output_stats=False,
                    keep_moving_average=True, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'stats':
            super(ExtBatchNorm, self).__init__(x, eps=eps, moving_fraction=moving_fraction,
                    external_stats=external_stats, output_stats=True,
                    keep_moving_average=False, use_moving_average=False,
                    name=name, update_collection=update_collection)
        elif phase == 'test':
            super(ExtBatchNorm, self).__init__(x, eps=eps,
                    external_stats=None, output_stats=False,
                    keep_moving_average=False, use_moving_average=True,
                    name=name, update_collection=update_collection)
        else:
            raise ValueError('INvalid phase: %s' % str(phase))

"""
class BatchNorm(LayerUpdateOps):
    def __init__(self, x, eps=1e-5, moving_fraction=0.9,
            keep_moving_average=True, use_self_stats=True,
            external_stats=None, output_stats=False,
            has_update_params=True, name='bn'):
        super(BatchNorm, self).__init__(name)
        assert has_update_params or not keep_moving_average, "Cannot keep moving average when the update params are not initialized."
        #assert is_training or not keep_moving_average, "Cannot keep moving average in test phase."
        self.inputs.append(x)
        self_stats = None
        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            params_shape = input_shape[-1:]
            reduce_axis = range(len(input_shape) - 1)
            # parameters for affine transformation
            beta = tf.get_variable('beta', params_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable('gamma', params_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(1.))
            self.params.extend([beta, gamma])
            # parameters used by update ops
            if has_update_params:
                moving_mean = tf.get_variable('moving_mean', params_shape, dtype=TF_DTYPE,
                        initializer=tf.constant_initializer(0.), trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, dtype=TF_DTYPE,
                        initializer=tf.constant_initializer(1.), trainable=False)
                mult_moving_mean = tf.get_variable('mult_moving_mean', [], dtype=TF_DTYPE,
                        initializer=tf.constant_initializer(1.), trainable=False)
                mult_moving_variance = tf.get_variable('mult_moving_variance', [], dtype=TF_DTYPE,
                        initializer=tf.constant_initializer(1.), trainable=False)
                self.update_params.extend([moving_mean, moving_variance,
                        mult_moving_mean, mult_moving_variance])
            # compute the statistics of itself
            if use_self_stats or output_stats or keep_moving_average:
                # variance correction factor
                n_var = np.prod(input_shape[:-1])
                var_factor = NP_DTYPE(1) if n_var == 1 else NP_DTYPE(n_var) / (n_var - 1)
                self_mean, self_variance = tf.nn.moments(x, reduce_axis)
                self_variance = var_factor * self_variance
                self_stats = [self_mean, self_variance]
            #
            if use_self_stats:
                mean, variance = self_stats
            elif external_stats is not None:
                assert len(external_stats) == 2, "external stats should contain two tensors (%d given)" % len(external_stats)
                self.inputs.extend(external_stats)
                mean, variance = external_stats
            else:
                mean, variance = (moving_mean / mult_moving_mean,
                        moving_variance / mult_moving_variance)    
            #
            if keep_moving_average:
                self_mean, self_variance = self_stats
                # update_ops can be regarded as a list of 4 tensors
                # To incorperate with control_dependencies, the update_ops are defined as the return values of a function
                self.update_ops.append(lambda : [
                        moving_mean.assign(moving_mean * moving_fraction + self_mean),
                        moving_variance.assign(moving_variance * moving_fraction + self_variance),
                        mult_moving_mean.assign(mult_moving_mean * moving_fraction + 1),
                        mult_moving_variance.assign(mult_moving_variance * moving_fraction + 1)]) 
            #
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps, name)
            self.outputs.append(x)
            if output_stats:
                self.outputs.extend(self_stats)
        self.print_info(LAYERS_VERBOSE)
"""
