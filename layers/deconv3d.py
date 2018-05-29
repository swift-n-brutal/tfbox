import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Deconv3d(Layer):
    """Transpose of 3D Convolutional layer

    """
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=-1, pad_type='CONSTANT',
            name='deconv3d', filler=('msra', 0., 1.)):
        """__init__ method of Deconv3d
        
        Parameters
        ----------
        x : tf.Tensor
            Input of 5D tensor.
        filter_shape : list of int
            Shape of convolutional filter. [filter_depth, filter_height, filter_width, chn_out]
            or [filter_depth, filter_height, filter_width, chn_out, chn_in]. As this layer is
            transposed version of conovlutional layer, the positions of input and
            output channels are swapped.
        bias : bool
            Whether to add bias.
        stride : int
            Stride size.
        pad_size : int
        pad_mode : str
            One of 'CONSTANT', 'REFLECT', or 'SYMMETRIC'
        name : string
        filler : tuple
            Initializer for convolutional weight. One of
            -   ('msra', negative_slope, positive_slope)
            -   ('gaussian', mean, stdev)
            -   ('uniform', minval, maxval)
        """                                                                      
        super(Deconv3d, self).__init__(name)
        self.inputs.append(x)
        input_shape = x.get_shape().as_list()
        if len(filter_shape) == 4:
            # get chn_in from input tensor
            kin = input_shape[-1]
            filter_shape.append(kin)
        kd, kh, kw, kout, kin = filter_shape
        # infer output shape
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
        #
        with tf.variable_scope(name):
            # initializer for weight
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
            # transposed convolution
            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv3d_transpose(x, weight, output_shape, [1, stride, stride, stride, 1], padding=padding)
            # add channel-wise bias
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
             # TODO Add pad_size feature in conv3d_transpose in future
            if pad_size > 0:
                x = tf.slice(x, [0, pad_size, pad_size, pad_size, 0],
                        size=[-1, output_shape[1]-pad_size*2, output_shape[2]-pad_size*2,
                            output_shape[3]-pad_size*2, -1])
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

