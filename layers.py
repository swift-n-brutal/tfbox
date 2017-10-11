import tensorflow as tf
#from tensorflow.python.ops import array_ops
import numpy as np
from config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE


class Layer(object):
    def __init__(self, name='base'):
        self.name = name
        self.inputs = list()
        self.outputs = list()
        self.params = list()
        
    def print_info(self, verbose):
        if verbose:
            print self.name
            print '\tIn', [(i.name, i.shape.as_list()) for i in self.inputs]
            print '\tOut', [(o.name, o.shape.as_list()) for o in self.outputs]

class LayerUpdateOps(Layer):
    def __init__(self, name='update_ops'):
        super(LayerUpdateOps, self).__init__(name)
        self.update_ops = list()
        self.update_params = list()

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

class Deconv3d(Layer):
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=-1, pad_type='CONSTANT',
            name='deconv3d', filler=('msra', 0., 1.)):
        super(Deconv3d, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            kd, kh, kw, kout, kin = filter_shape
            output_shape = [input_shape[0], 0, 0, 0, kout]
            if pad_size == -1:
                padding = 'SAME'
                output_shape[1] = input_shape[1] * stride
                output_shape[2] = input_shape[2] * stride
                output_shape[3] = input_shape[3] * stride
            elif pad_size >= 0:
                padding = 'VALID'
                output_shape[1] = (input_shape[1] - 1) * stride + kd
                output_shape[2] = (input_shape[2] - 1) * stride + kh
                output_shape[3] = (input_shape[3] - 1) * stride + kw
            else:
                raise ValueError('Cannot set nontrivial pad_size in Deconv3d: pad_size = %d' % pad_size)

            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                fan_in = kd * kh * kw * kin * 1.0 / (stride**2)
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))

            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv3d_transpose(x, weight, output_shape, [1, stride, stride, stride, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
            if pad_size > 0:
                x = tf.slice(x, [0, pad_size, pad_size, pad_size, 0],
                        size=[-1, output_shape[1]-pad_size*2, output_shape[2]-pad_size*2,
                            output_shape[3]-pad_size*2, -1])
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

class Conv2d(Layer):
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=0, pad_type='CONSTANT',
            name='conv2d', filler=('msra', 0., 1.)):
        super(Conv2d, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            kh, kw, kin, kout = filter_shape
            if pad_size == -1:
                pad_size = (kh - 1) / 2
                x = tf.pad(x, [[0,0], [pad_size,kh-1-pad_size], [pad_size,kh-1-pad_size], [0,0]], pad_type)
            elif pad_size > 0:
                x = tf.pad(x, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], pad_type)

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

            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding='VALID')

            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

class Deconv2d(Layer):
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=-1, pad_type='CONSTANT',
            name='deconv2d', filler=('msra', 0., 1.)):
        super(Deconv2d, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            kh, kw, kout, kin = filter_shape
            output_shape = [input_shape[0], 0, 0, kout]
            if pad_size == -1:
                padding = 'SAME'
                output_shape[1] = input_shape[1] * stride
                output_shape[2] = input_shape[2] * stride
            elif pad_size >= 0:
                padding = 'VALID'
                output_shape[1] = (input_shape[1] - 1) * stride + kh
                output_shape[2] = (input_shape[2] - 1) * stride + kw
            else:
                raise ValueError('Cannot set nontrivial pad_size in Deconv2d: pad_size = %d' % pad_size)

            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                fan_in = kh * kw * kin * 1.0 / (stride**2)
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))

            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv2d_transpose(x, weight, output_shape, [1, stride, stride, 1], padding=padding)
            
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
            if pad_size > 0:
                x = tf.slice(x, [0, pad_size, pad_size, 0],
                        size=[-1, output_shape[1]-pad_size*2, output_shape[2]-pad_size*2, -1])
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

class Linear(Layer):
    def __init__(self, x, output_dim, bias=True, name='fc', filler=('msra', 0., 1.)):
        super(Linear, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            fan_in = np.prod(shape[1:])
            x = tf.reshape(x, [-1, fan_in])
            
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

            weight = tf.get_variable('weight', shape=[fan_in, output_dim], dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.matmul(x, weight)

            if bias:
                b = tf.get_variable('bias', shape=[output_dim], dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

# Sampling layers

class Pool(Layer):
    def __init__(self, x, pool_type, ksize, stride=1, padding='VALID', name='pool'):
        super(Pool, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if pool_type == 'AVG':
                self.outputs.append(tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1], padding=padding))
            elif pool_type == 'MAX':
                self.outputs.append(tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                    strides=[1, stride, stride, 1], padding=padding))
            else:
                raise ValueError('Invalid pool type: %s' % (pool_type))
        self.print_info(LAYERS_VERBOSE)

class Pool3d(Layer):
    def __init__(self, x, pool_type, ksize, stride=1, padding='VALID', name='pool'):
        super(Pool3d, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if pool_type == 'AVG':
                self.outputs.append(tf.nn.avg_pool3d(x, ksize=[1, ksize, ksize, ksize, 1],
                    strides=[1, stride, stride, stride, 1], padding=padding))
            elif pool_type == 'MAX':
                self.outputs.append(tf.nn.max_pool3d(x, ksize=[1, ksize, ksize, ksize, 1],
                    strides=[1, stride, stride, stride, 1], padding=padding))
            else:
                raise ValueError('Invalid pool type: %s' % (pool_type))
        self.print_info(LAYERS_VERBOSE)

class NNUpsample(Layer):
    def __init__(self, x, factor, name='nnupsample'):
        super(NNUpsample, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            output_size = (shape[1] * factor, shape[2] * factor)
            self.outputs.append(tf.image.resize_nearest_neighbor(x, size=output_size))
        self.print_info(LAYERS_VERBOSE)

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
            self.outputs.append(tf.add_n(xs, name=name))
        self.print_info(LAYERS_VERBOSE)
            
class Concat(Layer):
    def __init__(self, xs, axis=-1, name='concat'):
        super(Concat, self).__init__(name)
        self.inputs.extend(xs)
        with tf.variable_scope(name):
            self.outputs.append(tf.concat(xs, axis, name=name))
        self.print_info(LAYERS_VERBOSE)

# Activation layers

class Act(Layer):
    def __init__(self, x, args, name='act'):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if args[0] == 'ReLU':
                if len(args) == 1:
                    self.ReLU(x)
                elif len(args) == 2:
                    self.ReLU(x, args[1])
            elif args[0] == 'ELU':
                if len(args) == 1:
                    self.ELU(x)
                elif len(args) == 2:
                    self.ELU(x, args[1])
            else:
                raise ValueError('Invalid activation type: %s' % str(args[0]))
        self.print_info(LAYERS_VERBOSE)

    def ELU(self, x, alpha=1., name='elu'):
        with tf.variable_scope(name):
            elu1 = tf.nn.elu(x)
            if alpha == 1.:
                self.outputs.append(elu1)
            else:
                #self.outputs.append(array_ops.where(x >= 0, elu1, alpha * elu1))
                self.outputs.append(tf.where(x >= 0, elu1, alpha * elu1))

    def ReLU(self, x, slope=0., name='relu'):
        with tf.variable_scope(name):
            if slope == 0.:
                self.outputs.append(tf.nn.relu(x))
            else:
                #self.outputs.append(array_ops.where(x >= 0, x, slope*x))
                self.outputs.append(tf.where(x >= 0, x, slope*x))

class BatchNorm(LayerUpdateOps):
    def __init__(self, x, eps=1e-5, moving_fraction=0.9,
            keep_moving_average=True, is_training=tf.constant(True), name='bn'):
        super(BatchNorm, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            params_shape = input_shape[-1:]
            reduce_axis = range(len(input_shape) - 1)

            beta = tf.get_variable('beta', params_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
            gamma = tf.get_variable('gamma', params_shape, dtype=TF_DTYPE, initializer=tf.constant_initializer(1.))
            self.params.extend([beta, gamma])
            if keep_moving_average:
                # variance correction factor
                n_var = np.prod(input_shape[:-1])
                var_factor = NP_DTYPE(1) if n_var == 1 else NP_DTYPE(n_var) / (n_var - 1)
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
                mean, variance = tf.cond(is_training,
                        lambda: tf.nn.moments(x, reduce_axis),
                        lambda: (moving_mean / mult_moving_mean,
                            moving_variance / mult_moving_variance))
                # update_ops can be regarded as a list of 4 tensors
                self.update_ops.extend(tf.cond(is_training,
                        lambda: (moving_mean.assign(moving_mean * moving_fraction + mean),
                            moving_variance.assign(moving_variance * moving_fraction + var_factor * variance),
                            mult_moving_mean.assign(mult_moving_mean * moving_fraction + 1),
                            mult_moving_variance.assign(mult_moving_variance * moving_fraction + 1)),
                        lambda: (moving_mean,
                            moving_variance,
                            mult_moving_mean,
                            mult_moving_variance)))
            else:
                mean, variance = tf.nn.moments(x, reduce_axis)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps, name)
            self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

# Loss layers

class LpLoss(Layer):
    def __init__(self, x, y, p, samplewise=False, name='lp_loss'):
        super(LpLoss, self).__init__(name)
        self.inputs.extend([x, y])
        with tf.variable_scope(name):
            if p != 1 and p != 2:
                raise ValueError('Unsupported LpLoss: p = %d' % p)
            if p == 1:
                eltwise = tf.abs(x-y)
            elif p == 2:
                eltwise = tf.square(x-y)
            if samplewise:
                shape = eltwise.get_shape().as_list()
                if len(shape) > 1:
                    self.outputs.append(tf.reduce_mean(eltwise, axis=shape[1:]))
                else:
                    self.outputs.append(eltwise)
            else:
                self.outputs.append(tf.reduce_mean(eltwise))
        self.print_info(LAYERS_VERBOSE)

class SumLoss(Layer):
    def __init__(self, x, mean=True, name='sum_loss'):
        super(SumLoss, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if mean:
                self.outputs.append(tf.reduce_mean(x))
            else:
                self.outputs.append(tf.reduce_sum(x))
        self.print_info(LAYERS_VERBOSE)

class Accuracy(Layer):
    def __init__(self, logit, label, topk=1, name='accuracy'):
        super(Accuracy, self).__init__(name)
        self.inputs.extend([logit, label])
        with tf.variable_scope(name):
            assert topk == 1, 'Invalid parameter (topk = %s)' % str(topk)
            correct_prediction = tf.equal(
                    tf.cast(tf.argmax(logit, 1), tf.int32), tf.cast(label, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.outputs.append(accuracy)
        self.print_info(LAYERS_VERBOSE)
