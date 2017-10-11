# TODO
import numpy as np
import tensorflow as tf
import layers as l
from model_base import ModelBase

class GeneratorDCGAN(ModelBase):
    def __init__(self, code_dim=128, batchsize=64, output_chn=3, output_size=64,
            depth=5, repeat=1, chn=32, kernel_size=3, stride=2,
            filler=('msra', 0., 1.), bn=False, act=('ReLU', 0.), deconv=True, name='dcgan/gen'):
        super(GeneratorDCGAN, self).__init__()
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
        self.act = act
        self.deconv = deconv
        self.name = name
        self._init = True
        
        # setup
        with tf.variable_scope(name):
            input_tensor = tf.placeholder(tf.float32, [batchsize, code_dim], name='rand_vec')
        self.inputs.append(input_tensor)
        tops = self.run(self.inputs)
        self.outputs.extend(tops)
        self._init = False

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
        act = self.act
        deconv = self.deconv
        name = self.name
        init = self._init
        input_tensor = bottoms[0]
        outputs = list()

        print
        print '===== Setup', input_tensor.name.rsplit('/', 1)[0], '====='
        print 
        with tf.variable_scope(name, reuse=(not init)):
            # compute dimension of decoder
            chns = range(depth)
            chns[depth-1] = chn
            first_feat_size = output_size
            for i in xrange(1, depth):
                chns[depth-i-1] = chns[depth-i] * 2
                first_feat_size = first_feat_size / stride
            first_feat_shape = [-1, first_feat_size, first_feat_size, chns[0]]
            first_feat_dim = np.prod(first_feat_shape[1:])
    
            # fc decode
            tops = self._append(l.Linear(input_tensor, first_feat_dim, name='fc_decode', filler=filler), init=init)
            tops = [tf.reshape(tops[0], first_feat_shape)]
    
            # conv layers + nn upsample
            filter_shape = [kernel_size, kernel_size, chns[0], chns[0]]
            for i in xrange(depth):
                filter_shape[3] = chns[i]
                if i > 0:
                    if deconv:
                        tops = self._append(
                                l.Deconv2d(tops[0], [stride, stride, chns[i], chns[i-1]],
                                        stride=stride, name='deconv%d' % (i-depth), filler=filler), init=init)
                        filter_shape[2] = chns[i]
                    else:
                        tops = self._append(l.NNUpsample(tops[0], stride), init=init)
                for j in xrange(repeat):
                    tops = self._append(l.Conv2d(tops[0], filter_shape, pad_size=-1,
                            name='conv%d_%d' % (i-depth, j), filler=filler), init=init)
                    tops = self._append(l.Act(tops[0], act))
                    if j == 0:
                        filter_shape[2] = chns[i]
    
            # output image
            filter_shape[3] = output_chn
            tops = self._append(l.Conv2d(tops[0], filter_shape, pad_size=-1, name='conv_output', filler=filler), init=init)
            outputs.extend(tops)
        return outputs

class DiscriminatorDCGAN(ModelBase):
    def __init__(self, code_dim=128, batchsize=64, input_chn=3, input_size=64, input_depth=5,
            repeat=1, chn=32, kernel_size=3, stride=2, output_chn=3, output_size=64, output_depth=64,
            filler=('msra', 0., 1.), bn=False, act=('ReLU', 0.2), deconv=True, name='dcgan/dis'):
        super(DiscriminatorDCGAN, self).__init__()
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
        self.act = act
        self.deconv = deconv
        self.name = name
        self._init = True
        
        # setup
        with tf.variable_scope(name):
            input_tensor = tf.placeholder(tf.float32, [batchsize, input_size, input_size, input_chn], name='input')
        self.inputs.append(input_tensor)
        tops = self.run(self.inputs)
        self.outputs.extend(tops)
        self._init = False

    def run(self, bottoms):
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
        act = self.act
        deconv = self.deconv
        name = self.name
        init = self._init
        input_tensor = bottoms[0]
        outputs = list()

        print
        print '===== Setup', input_tensor.name.rsplit('/', 1)[0], '====='
        print
        with tf.variable_scope(name, reuse=(not init)):
            # conv input
            filter_shape = [kernel_size, kernel_size, input_chn, chn]
            tops = self._append(l.Conv2d(input_tensor, filter_shape, pad_size=-1,
                    name='conv_input', filler=filler), init=init)
            tops = self._append(l.Act(tops[0], act))
    
            # compute dimensions of encoder
            depth = input_depth
            chns = range(depth)
            chns[0] = chn
            for i in xrange(1, depth):
                chns[i] = chns[i-1]*2
    
            # conv layers + downsample
            for i in xrange(depth):
               if i > 0:
                   tops = self._append(l.Conv2d(tops[0], [stride, stride, chns[i-1], chns[i]],
                           stride=stride, name='downsample%d' % (i), filler=filler), init=init)
                   tops = self._append(l.Act(tops[0], act))
               filter_shape[2] = filter_shape[3] = chns[i]
               for j in xrange(repeat):
                   tops = self._append(l.Conv2d(tops[0], filter_shape, pad_size=-1,
                           name='conv%d_%d' % (i, j), filler=filler), init=init)
                   tops = self._append(l.Act(tops[0], act))
               
            # fc encoder
            tops = self._append(l.Linear(tops[0], 1, name='fc_score', filler=filler, bias=False), init=init)
            outputs.extend(tops)
            
        return outputs
