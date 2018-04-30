import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class SoftmaxLoss(Layer):
    def __init__(self, logits, labels, samplewise=False, name='softmax_loss'):
        super(SoftmaxLoss, self).__init__(name)
        self.inputs.extend([logits, labels])
        with tf.variable_scope(name):
            sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
            if samplewise:
                self.outputs.append(sample_loss)
            else:
                self.outputs.append(tf.reduce_mean(sample_loss))
        self.print_info(LAYERS_VERBOSE)

