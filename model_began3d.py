import numpy as np
import tensorflow as tf
import layers as l
from model_base import ModelBase

class Generator(ModelBase):
    def __init__(self, code_dim=128, batchsize=16, output_chn=3, output_size=64,
            depth=4, repeat=2, chn=16, kernel_size=3, stride=2,
            filler=('msra', 0., 1.), bn=False, alpha=1.0, deconv=False, inputs=None, name='began/gen'):
        super(Generator, self).__init__()
        self.code_dim = code_dim
        self.batchsize = batchsize
        self.output_chn = output_chn
        self.output_size = output_size
        self.depth = depth
        self.repeat = repeat
        self.chn = chn
        self.kernel_size = kernel_size
        self.stride = stride
        self.filler = filler
        self.bn = bn
        self.alpha = alpha
        self.deconv = deconv
        self.name = name
        
        # setup
        with tf.variable_scope(name):
            if inputs is None:
                input_tensor = tf.placeholder(tf.float32, [batchsize, code_dim], name='rand_vec')
                self.inputs.append(input_tensor)
            else:
                self.inputs.extend(inputs)

        self._init = True
        tops = self.run(self.inputs)
        self._init = False
        self.outputs.extend(tops)

    def run(self, bottoms):
        code_dim = self.code_dim
        batchsize = self.batchsize
        output_chn = self.output_chn
        output_size = self.output_size
        depth = self.depth
        repeat = self.repeat
        chn = self.chn
        kernel_size = self.kernel_size
        stride = self.stride
        filler = self.filler
        bn = self.bn
        alpha = self.alpha
        deconv = self.deconv
        name = self.name
        init = self._init
        input_tensor = bottoms[0]
        outputs = list()

        print
        print '===== Setup', name, 'from', input_tensor.name, '====='
        print
        with tf.variable_scope(name, reuse=(not init)):
            # compute dimension of decoder
            chns = range(depth)
            chns[depth-1] = chn
            first_feat_size = output_size
            for i in xrange(1, depth):
                chns[depth-i-1] = chns[depth-i] * 2
                first_feat_size = first_feat_size / stride
            first_feat_shape = [-1, first_feat_size, first_feat_size, first_feat_size, chns[0]]
            first_feat_dim = np.prod(first_feat_shape[1:])
    
            # fc decode
            tops = self._append(l.Linear(input_tensor, first_feat_dim, name='fc_decode', filler=filler), init=init)
            tops = [tf.reshape(tops[0], first_feat_shape)]
    
            # conv layers + nn upsample
            filter_shape = [kernel_size, kernel_size, kernel_size, chns[0], chns[0]]
            for i in xrange(depth):
                if i > 0:
                    if deconv:
                        tops = self._append(
                                l.Deconv3d(tops[0], [stride, stride, stride, chns[i], chns[i-1]],
                                        stride=stride, name='deconv%d' % (i-depth), filler=filler), init=init)
                        filter_shape[-2] = chns[i]
                    else:
                        tops = self._append(l.NNUpsample(tops[0], stride), init=init)
                filter_shape[-1] = chns[i]
                for j in xrange(repeat):
                    tops = self._append(l.Conv3d(tops[0], filter_shape, pad_size=-1,
                            name='conv%d_%d' % (i-depth, j), filler=filler), init=init)
                    tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
                    if j == 0:
                        filter_shape[-2] = chns[i]
    
            # output image
            filter_shape[-1] = output_chn
            tops = self._append(l.Conv3d(tops[0], filter_shape, pad_size=0, name='conv_output', filler=filler), init=init)
            outputs.extend(tops)
        return outputs

class Discriminator(ModelBase):
    def __init__(self, code_dim=512, batchsize=64, input_chn=1, input_size=30, input_depth=4,
            repeat=1, chn=64, kernel_size=3, stride=2, output_chn=1, output_size=32, output_depth=4,
            filler=('gaussian', 0., 0.02), bn=False, alpha=1.0, deconv=True, inputs=None, name='began/dis'):
        super(Discriminator, self).__init__()
        self.code_dim = code_dim
        self.batchsize = batchsize
        self.input_chn = input_chn
        self.input_size = input_size
        self.input_depth = input_depth
        self.output_chn = output_chn
        self.output_size = output_size
        self.output_depth = output_depth
        self.repeat = repeat
        self.chn = chn
        self.kernel_size = kernel_size
        self.stride = stride
        self.filler = filler
        self.bn = bn
        self.alpha = alpha
        self.deconv = deconv
        self.name = name
        
        # setup
        self._init = True
        tops = self.run(inputs)
        self._init = False
        self.outputs.extend(tops)

    def run(self, inputs):
        code_dim = self.code_dim
        batchsize = self.batchsize
        input_chn = self.input_chn
        input_size = self.input_size
        input_depth = self.input_depth
        output_chn = self.output_chn
        output_size = self.output_size
        output_depth = self.output_depth
        repeat = self.repeat
        chn = self.chn
        kernel_size = self.kernel_size
        stride = self.stride
        filler = self.filler
        bn = self.bn
        alpha = self.alpha
        deconv = self.deconv
        name = self.name
        init = self._init
        outputs = list()

        with tf.variable_scope(name, reuse=(not init)):
            if init:
                self.is_training = tf.placeholder(tf.bool, [], name='is_training')
                if inputs is None:
                    inputs = [tf.placeholder(tf.float32,
                            [batchsize, input_size, input_size, input_size, input_chn], name='input')]
                self.inputs.extend(inputs)
            input_tensor = inputs[0]
            print
            print '===== Setup', name, 'from', input_tensor.name, '====='
            print
            # conv input
            filter_shape = [kernel_size, kernel_size, kernel_size, input_chn, chn]
            tops = self._append(l.Conv3d(input_tensor, filter_shape, pad_size=2,
                    name='conv_input', filler=filler), init=init)
            tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
            filter_shape[-2] = chn
    
            # compute dimensions of encoder
            depth = input_depth
            chns = range(depth)
            chns[0] = chn
            for i in xrange(1, depth):
                #chns[i] = chns[i-1]*2
                chns[i] = chns[i-1] + chn
    
            # conv layers + downsample
            for i in xrange(depth):
                if i > 0:
                    if deconv:
                        tops = self._append(l.Conv3d(tops[0], [stride, stride, stride, chns[i-1], chns[i]],
                                stride=stride, name='conv%d_down' % (i), filler=filler), init=init)
                        tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
                        filter_shape[-2] = chns[i]
                    else:
                        tops = self._append(l.Pool3d(tops[0], 'AVG', stride, stride, name='pool%d_down' % i), init=init)
                filter_shape[-1] = chns[i]
                for j in xrange(repeat):
                    tops = self._append(l.Conv3d(tops[0], filter_shape, pad_size=-1,
                            name='conv%d_%d' % (i, j), filler=filler), init=init)
                    tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
                    if j == 0:
                        filter_shape[-2] = chns[i]
               
            # fc encoder
            tops = self._append(l.Linear(tops[0], code_dim, name='fc_encode', filler=filler), init=init)
    #        self.outputs.extend(tops)
            outputs.extend(tops)
            
            # compute dimension of decoder
            depth = output_depth
            chns = range(depth)
            chns[depth-1] = chn
            first_feat_size = output_size
            for i in xrange(1, depth):
                #chns[depth-i-1] = chns[depth-i] * 2
                chns[depth-i-1] = chns[depth-i] + chn
                first_feat_size = first_feat_size / stride
            first_feat_shape = [-1, first_feat_size, first_feat_size, first_feat_size, chns[0]]
            first_feat_dim = np.prod(first_feat_shape[1:])
    
            # fc decode
            tops = self._append(l.Linear(tops[0], first_feat_dim, name='fc_decode', filler=filler), init=init)
            tops = [tf.reshape(tops[0], first_feat_shape)]
    
            # conv layers + nn upsample
            filter_shape = [kernel_size, kernel_size, kernel_size, chns[0], chns[0]]
            for i in xrange(depth):
                if i > 0:
                    if deconv:
                        tops = self._append(
                                l.Deconv3d(tops[0], [stride, stride, stride, chns[i], chns[i-1]],
                                        stride=stride, name='deconv%d' % (i-depth), filler=filler), init=init)
                        tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
                        filter_shape[-2] = chns[i]
                    else:
                        tops = self._append(l.NNUpsample(tops[0], stride, name='nnup%d' % (i-depth)), init=init)
                filter_shape[-1] = chns[i]
                for j in xrange(repeat):
                    tops = self._append(l.Conv3d(tops[0], filter_shape, pad_size=-1,
                            name='conv%d_%d' % (i-depth, j), filler=filler), init=init)
                    tops = self._append(l.Act(tops[0], ('ELU', alpha)), init=init)
                    if j == 0:
                        filter_shape[-2] = chns[i]
    
            # output image
            filter_shape[-1] = output_chn
            tops = self._append(l.Conv3d(tops[0], filter_shape, pad_size=0, name='conv_output', filler=filler), init=init)
    #        self.outputs.extend(tops)
            outputs.extend(tops)
        return outputs
