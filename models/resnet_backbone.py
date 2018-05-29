import numpy as np
import tensorflow as tf
import tfbox.layers as L
from .model import Model
from ..config import TF_DTYPE

# model definitions

class ResNetBackbone(object):
    """Backbone architecture of residual network.

    A residual network consists of three parts: 'base', 'body' and 'head'.
    The 'base' part is a connection between the input and the 'body' part.
    The 'body' part consists of several sets of residual blocks.
    The 'head' part is a connection between the 'body' part and the output.

    Attributes
    ----------
    blocks : list of int
        Number of residual blocks in len(blocks) stages.
    chns : list of int
        Dimension of channel of each group. This should have the same length
        with that of blocks.
    strides : list of int
        Stride size of each group. This should have the same length with that
        of blocks.
    filler : tuple
        This specifies how to initialize weights in weighted layers.
    act_use_bn : bool
        Whether or not to use batch norm.
    act_nonlinear : tuple
        Nonlinear activation function.
    bottleneck : bool
        Whether or not to use bottleneck residual block.
    name : str
        Name of the architecture. Also specifies the variable scope for
        this architecture.
    """
    def __init__(self, blocks, chns, strides,
            kernel_size, filler,
            act_use_bn, act_nonlinear,
            bottleneck=False, name='resnet_backbone'):
        self.name = name
        #
        self.blocks = blocks
        self.chns = chns
        self.strides = strides
        self.act_use_bn = act_use_bn
        self.act_nonlinear = act_nonlinear
        self.kernel_size = kernel_size
        self.filler = filler
        self.bottleneck = bottleneck

    def build_model(self, inputs, name, phase, model=None,
            condition=None, update_collection=None):
        if model is None:
            model = Model()
            model.inputs.extend(inputs)
        #
        blocks = self.blocks
        chns = self.chns
        strides = self.strides
        kernel_size = self.kernel_size
        filler = self.filler
        bottleneck = self.bottleneck
        act_config = {
                'condition': condition,
                'phase': phase,
                'update_collection': update_collection,
                'use_bn': self.act_use_bn,
                'nonlinear': self.act_nonlinear}
        #
        with tf.name_scope(name):
            # Set reuse=tf.AUTO_REUSE assuming that this is the topmost level of variable scope
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # append base
                tops = self.append_base(model, inputs, act_config)
                # append resblocks
                for stage, n_blocks in enumerate(blocks):
                    chn = chns[stage]
                    stride = strides[stage]
                    tops = self.append_resblocks(model, tops, n_blocks, chn, stride,
                            act_config, kernel_size, filler, bottleneck, name='s%d' % stage)
                # append head
                tops = self.append_head(model, tops, act_config)
        #
        model.outputs.extend(tops)
        return model

    @classmethod
    def append_act(cls, model, inputs, config, name='act'):
        with tf.variable_scope(name):
            tops = inputs
            # Normalization
            if config['use_bn']:
                if config.get('condition') is None:
                    # batch norm
                    tops = model.append(L.BatchNorm(
                        tops[-1], phase=config['phase'],
                        name='bn', update_collection=config['update_collection']))
                else:
                    # conditional batch norm
                    tops = model.append(L.CondBatchNorm(
                        tops[-1], config['condition'],
                        phase=config['phase'], name='cond_bn',
                        update_collection=config['update_collection']))
            # Non-linear
            tops = model.append(L.Act(tops[-1], config['nonlinear']))
        return tops

    @classmethod
    def append_resblocks(cls, model, inputs, n_blocks, chn, stride, act_config,
            kernel_size=3, filler=('msra', 0., 1.), bottleneck=False, name='stage'):
        tops = inputs
        with tf.variable_scope(name):
            for b in xrange(n_blocks):
                tops = cls.append_resblock(model, tops, chn,
                        stride if b == 0 else 1,
                        act_config, kernel_size, filler, both_act=(b == 0),
                        bottleneck=bottleneck, name='b%d' % b)
        return tops

    @classmethod
    def append_resblock(cls, model, inputs, chn, stride, act_config,
            kernel_size=3, filler=('msra', 0., 1.),
            both_act=False, bottleneck=False, name='block'):
        chn_in = inputs[-1].shape.as_list()[-1]
        add_bias = not act_config['use_bn']
        is_training = act_config['phase'] == 'train'
        update_collection = act_config['update_collection']
        with tf.variable_scope(name):
            tops = cls.append_act(model, inputs, act_config, 'act_pre')
            if both_act:
                shortcut_top = tops[-1]
            else:
                shortcut_top = inputs[-1]
            if bottleneck:
                # [1, 1, chn or chn_in / 4] -> [ks, ks, chn]/stride -> [1, 1, chn * 4]
                # When appended right after the base part, the input channel may not
                # follow the rule. We assume that the stride size of the residual
                # block after the base part is always 1. TODO Add a switch. 
                chn_out = chn*4
                chn_conv1 = chn if stride == 1 else chn_in / 4 
                tops = model.append(
                        L.Conv2d(tops[-1], [1, 1, chn_conv1], bias=add_bias, name='conv1', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act1')
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, chn], bias=add_bias,
                            stride=stride, pad_size=-1, name='conv2', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act2')
                tops = model.append(
                        L.Conv2d(tops[-1], [1, 1, chn_out], bias=add_bias, name='conv3', filler=filler))
            else:
                # [ks, ks, chn]/stride  -> [ks, ks, chn]
                chn_out = chn
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, chn], bias=add_bias,
                            stride=stride, pad_size=-1, name='conv1', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act1')
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, chn_out], bias=add_bias,
                            pad_size=-1, name='conv2', filler=filler))
            if chn_in != chn_out or stride > 1:
                # projection to match output shapes
                shortcut_top = model.append(
                        L.Conv2d(shortcut_top, [stride, stride, chn_out], bias=add_bias,
                            stride=stride, name='proj', filler=filler))[-1]
            tops = model.append(L.Add([shortcut_top, tops[-1]]))
        return tops
    
    def append_base(self, model, inputs, act_config, name='base'):
        print "Resnet(Base): Use identity mapping here. This should be implemented in specific model definitions."
        return inputs

    def append_head(self, model, inputs, act_config, name='head'):
        print "Resnet(Head): Use identity mapping here. This should be implemented in specific model definitions."
        return inputs

class DeconvResNetBackbone(ResNetBackbone):
    @classmethod
    def append_resblock(cls, model, inputs, chn, stride, act_config,
            kernel_size=3, filler=('msra', 0., 1.),
            both_act=False, bottleneck=False, name='block'):
        chn_in = inputs[-1].shape.as_list()[-1]
        add_bias = not act_config['use_bn']
        is_training = act_config['phase'] == 'train'
        update_collection = act_config['update_collection']
        with tf.variable_scope(name):
            tops = cls.append_act(model, inputs, act_config, 'act_pre')
            if both_act:
                shortcut_top = tops[-1]
            else:
                shortcut_top = inputs[-1]
            if bottleneck:
                # [1, 1, chn or chn_in / 4] -> [ks, ks, chn]*stride -> [1, 1, chn * 4]
                # When appended right after the base part, the input channel may not
                # follow the rule. We assume that the stride size of the residual
                # block after the base part is always 1. TODO Add a switch. 
                chn_out = chn*4
                chn_conv1 = chn if stride == 1 else chn_in / 4
                tops = model.append(
                        L.Conv2d(tops[-1], [1, 1, chn_conv1], bias=add_bias, name='conv1', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act1')
                tops = model.append(
                        L.Deconv2d(tops[-1], [kernel_size, kernel_size, chn], bias=add_bias,
                            stride=stride, name='deconv2', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act2')
                tops = model.append(
                        L.Conv2d(tops[-1], [1, 1, chn_out], bias=add_bias, name='conv3', filler=filler))
            else:
                # [ks, ks, chn]*stride -> [ks, ks, chn]
                chn_out = chn
                tops = model.append(
                        L.Deconv2d(tops[-1], [kernel_size, kernel_size, chn], bias=add_bias,
                            stride=stride, pad_size=-1, name='deconv1', filler=filler))
                tops = cls.append_act(model, tops, act_config, 'act1')
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, chn_out], bias=add_bias,
                            pad_size=-1, name='conv2', filler=filler))
        
