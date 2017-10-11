import numpy as np
import tensorflow as tf
import layers as l
from model_base import ModelBase
from config import TF_DTYPE as DTYPE

class Classifier(ModelBase):
    def __init__(self, n_classes, batch_size, vox_size=30, chn=32, input_chn=1,
            filler=('gaussian', 0., 0.02), inputs=None, is_training=None, name='classifier'):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.chn = chn
        self.input_chn = input_chn
        self.vox_size = vox_size
        self.batch_size = batch_size
        self.shape = [batch_size, vox_size, vox_size, vox_size, input_chn]
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
        n_classes = self.n_classes
        chn = self.chn
        input_chn = self.input_chn
        vox_size = self.vox_size
        batch_size = self.batch_size
        filler = self.filler
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
            # conv1 <- (30, 30, 30, 1)
            filter_shape = [5, 5, 5, input_chn, chn]
            tops = self._append(l.Conv3d(input_tensor, filter_shape, bias=True,
                    stride=2, pad_size=1, name='conv1', filler=filler), init=init)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.1)), init=init)
            feature_shape = tops[0].get_shape().as_list()
            noise_shape = [feature_shape[0], 1, 1, 1, feature_shape[-1]]
            tops = [tf.cond(is_training,
                    lambda: tf.nn.dropout(tops[0], tf.constant(0.8, dtype=DTYPE),
                        noise_shape=tf.constant(noise_shape, dtype=tf.int32)),
                    lambda: tops[0])]
            # conv2 <- (14, 14, 14, chn)
            filter_shape = [3, 3, 3, chn, chn]
            tops = self._append(l.Conv3d(tops[0], filter_shape, bias=True,
                    stride=1, pad_size=0, name='conv2', filler=filler), init=init)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.1)), init=init)
            # maxpool <- (12, 12, 12, chn)
            tops = self._append(l.Pool3d(tops[0], 'MAX', ksize=2, stride=2, name='maxpool'), init=init)
            feature_shape = tops[0].get_shape().as_list()
            noise_shape = [feature_shape[0], 1, 1, 1, feature_shape[-1]]
            tops = [tf.cond(is_training,
                    lambda: tf.nn.dropout(tops[0], tf.constant(0.7, dtype=DTYPE),
                        noise_shape=tf.constant(noise_shape, dtype=tf.int32)),
                    lambda: tops[0])]
            # fc1 <- (6, 6, 6, chn)
            tops = self._append(l.Linear(tops[0], 128, bias=True, name='fc1', filler=filler), init=init)
            tops = self._append(l.Act(tops[0], ('ReLU', 0.)), init=init)
            noise_shape = tops[0].get_shape().as_list()
            tops = [tf.cond(is_training,
                    lambda: tf.nn.dropout(tops[0], tf.constant(0.6, dtype=DTYPE),
                        noise_shape=tf.constant(noise_shape, dtype=tf.int32)),
                    lambda: tops[0])]
            # fc2 <- (128)
            tops = self._append(l.Linear(tops[0], n_classes, bias=True, name='fc2', filler=filler), init=init)
            outputs.append(tops[0])
        return outputs

