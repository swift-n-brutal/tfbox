import numpy as np
import tensorflow as tf
import layers as l
from model_base import ModelBase

class Unet(ModelBase):
    def __init__(self, batch_size, vox_size=30, chn=64, input_chn=1, sigmoid=True,
            filler=('gaussian', 0., 0.02), inputs=None, is_training=None, name='unet'):
        super(Unet, self).__init__()
        self.chn = chn
        self.input_chn = input_chn
        self.vox_size = vox_size
        self.batch_size = batch_size
        self.shape = [batch_size, vox_size, vox_size, vox_size, input_chn]
        self.sigmoid = sigmoid
        self.filler = filler
        self.name = name
        self.is_training = is_training

        self._init = True
        tops = self.run(inputs, update='init')
        self._init = False
        self.outputs.extend(tops)

    def run(self, inputs, update=None):
        name = self.name
        shape = self.shape
        init = self._init
        chn = self.chn
        input_chn = self.input_chn
        vox_size = self.vox_size
        batch_size = self.batch_size
        filler = self.filler
        sigmoid = self.sigmoid
        outputs = list()

        with tf.variable_scope(name, reuse=(not init)):
            if init:
                if self.is_training is None:
                    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
                if inputs is None:
                    inputs = [tf.placeholder(tf.float32, self.shape, name='input')]
                self.inputs.extend(inputs)
            is_training = self.is_training
            input_tensor = inputs[0]
            print
            print '===== Setup', name, 'from', input_tensor.name, '====='
            print
            # enc1 <- (30, 30, 30, 1)
            filter_shape = [4, 4, 4, input_chn, chn]
            tops = self._append(l.Conv3d(input_tensor, filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc1', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc1'),
                    init=init, update=update)
            bn_enc1 = tops[0]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # enc2 <- (15, 15, 15, chn)
            filter_shape = [4, 4, 4, chn, chn*2]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc2', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc2'),
                    init=init, update=update)
            bn_enc2 = tops[0]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # enc3 <- (7, 7, 7, chn*2)
            filter_shape = [4, 4, 4, chn*2, chn*4]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc3', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc3'),
                    init=init, update=update)
            bn_enc3 = tops[0]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # enc4 <- (3, 3, 3, chn*4)
            filter_shape = [3, 3, 3, chn*4, chn*8]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=1, pad_size=0, name='enc4', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc4'),
                    init=init, update=update)
            bn_enc4 = tops[0]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.)), init=init)
            # dec1 <- (1, 1, 1, chn*8)
            filter_shape = [3, 3, 3, chn*4, chn*8]
            tops = self._append(l.Deconv3d(tops[0], filter_shape, bias=False,
                    stride=1, pad_size=0, name='dec1', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_dec1'),
                    init=init, update=update)
            bn_dec1 = tops[0]
            tops = [tf.concat([bn_dec1, bn_enc3], axis=4)]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.)), init=init)
            # dec2 <- (3, 3, 3, chn*4*2)
            filter_shape = [3, 3, 3, chn*2, chn*4*2]
            tops = self._append(l.Deconv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=0, name='dec2', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_dec2'),
                    init=init, update=update)
            bn_dec2 = tops[0]
            tops = [tf.concat([bn_dec2, bn_enc2], axis=4)]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.)), init=init)
            # dec3 <- (7, 7, 7, chn*2*2)
            filter_shape = [3, 3, 3, chn, chn*2*2]
            tops = self._append(l.Deconv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=0, name='dec3', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_dec3'),
                    init=init, update=update)
            bn_dec3 = tops[0]
            tops = [tf.concat([bn_dec3, bn_enc1], axis=4)]
            tops = self._append(l.Act(tops[0], ('ReLU', 0.)), init=init)
            # dec4 <- (15, 15, 15, chn*1*2)
            filter_shape = [4, 4, 4, input_chn, chn*2]
            tops = self._append(l.Deconv3d(tops[0], filter_shape, bias=True,
                    stride=2, pad_size=-1, name='dec4', filler=filler), init=init)
            # output <- (30, 30, 30, 1)
            if sigmoid:
                outputs.append(tf.sigmoid(tops[0]))
            else:
                outputs.append(tops[0])
        return outputs

class Dnet(ModelBase):
    def __init__(self, batch_size, vox_size=30, chn=64, input_chn=1,
            filler=('gaussian', 0., 0.02), inputs=None, name='dnet'):
        super(Dnet, self).__init__()
        self.chn = chn
        self.input_chn = input_chn
        self.vox_size = vox_size
        self.batch_size = batch_size
        self.shape = [batch_size, vox_size, vox_size, vox_size, input_chn]
        self.filler = filler
        self.name = name

        self._init = True
        tops = self.run(inputs, update='init')
        self._init = False
        self.outputs.extend(tops)

    def run(self, inputs, update=None):
        name = self.name
        shape = self.shape
        init = self._init
        chn = self.chn
        input_chn = self.input_chn
        vox_size = self.vox_size
        batch_size = self.batch_size
        filler = self.filler
        outputs = list()

        with tf.variable_scope(name, reuse=(not init)):
            if init:
                self.is_training = tf.placeholder(tf.bool, [], name='is_training')
                if inputs is None:
                    inputs = [tf.placeholder(tf.float32, self.shape, name='input')]
                self.inputs.extend(inputs)
            is_training = self.is_training
            input_tensor = inputs[0]
            print
            print '===== Setup', name, 'from', input_tensor.name, '====='
            print
            # enc1 <- (30, 30, 30, 1)
            filter_shape = [4, 4, 4, input_chn, chn]
            tops = self._append(l.Conv3d(input_tensor, filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc1_1', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc1_1'),
                    init=init, update=update)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            filter_shape = [3, 3, 3, chn, chn]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=1, pad_size=1, name='enc1_2', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc1_2'),
                    init=init, update=update)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # enc2 <- (15, 15, 15, 64)
            filter_shape = [3, 3, 3, chn, chn*2]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc2_1', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc2_1'),
                    init=init, update=update)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            filter_shape = [3, 3, 3, chn*2, chn*2]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=1, pad_size=1, name='enc2_2', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc2_2'),
                    init=init, update=update)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # enc3 <- (8, 8, 8, 128)
            filter_shape = [3, 3, 3, chn*2, chn*4]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=False,
                    stride=2, pad_size=1, name='enc3', filler=filler), init=init)
            tops = self._append(l.BatchNorm(tops[0], is_training=is_training, name='bn_enc3'),
                    init=init, update=update)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.2)), init=init)
            # fc <- (4, 4, 4, 256)i
            tops = self._append(l.Linear(tops[0], 2, bias=True, filler=filler), init=init)
            outputs.extend(tops)
        return outputs
