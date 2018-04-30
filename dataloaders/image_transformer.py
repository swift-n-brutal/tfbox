# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 17:15:03 2017

@author: shiwu_001
"""

import os
import os.path as osp
import numpy as np
from scipy import misc, ndimage
from ..config import NP_DTYPE as DTYPE

class ImageTransformer(object):
    def __init__(self, inputs, seed=None):
#        super(ImageTransformer, self).__init__(seed=seed)
        assert type(inputs) == dict, 'Tranformer.inputs = {blob_name: blob_shape}'
        self.rand = np.random.RandomState(seed)
        self.inputs = inputs
        self.scale = {}
        self.pad = {}
        self.pad_value = {}
        self.mean = {}
        self.std = {}
        self.mirror = {}
        self.center = {}
        self.transpose = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception("{} is not one of the net inputs: {}".format(
                            in_, self.inputs))

    def add_input(self, name, shape):
        if name in self.inputs:
            raise Exception("%s is already set as an input" % name)
        self.inputs[name] = shape
            
    def set_scale(self, in_, scale):
        self.__check_input(in_)
        if scale.shape == () or scale.shape == (1,):
            scale = scale.reshape(1)
        elif scale.shape != (2, 1) and scale.shape != (2,) and scale.shape != (2,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale[in_] = scale
            
    def set_mean(self, in_, mean):
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][-1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[np.newaxis, np.newaxis, :]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = ms + (1,)
            if len(ms) != 3:
                raise ValueError('Mean shape invalid.')
            if ms != self.inputs[in_][1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_std(self, in_, std):
        self.__check_input(in_)
        ss = std.shape
        if std.ndim == 1:
            # broadcast channels
            if ss[0] != self.inputs[in_][-1]:
                raise ValueError('Std channels incompatible with input.')
            std = std[np.newaxis, np.newaxis, :]
        else:
            # elementwise std
            if len(ss) == 2:
                ss = ss + (1,)
            if len(ss) != 3:
                raise ValueError('Std shape invalid.')
            if ss != self.inputs[in_][1:]:
                raise ValueError('Std shape incompatible with input shape.')
        self.std[in_] = std
    
    def set_pad(self, in_, pad):
        self.__check_input(in_)
        self.pad[in_] = pad

    def set_pad_value(self, in_, pad_value):
        self.__check_input(in_)
        self.pad_value[in_] = pad_value

    def set_mirror(self, in_, mirror):
        self.__check_input(in_)
        self.mirror[in_] = mirror

    def set_center(self, in_, center):
        self.__check_input(in_)
        self.center[in_] = center

    def set_transpose(self, in_, transpose):
        self.__check_input(in_)
        self.transpose[in_] = transpose

    def process(self, in_, data, data_format='HWC'):
        self.__check_input(in_)
        if data_format == 'HWC':
            data_in = np.copy(data).astype(DTYPE)
        elif data_format == 'CHW':
            data_in = np.copy(np.transpose(data, (1, 2, 0))).astype(DTYPE)
        else:
            raise ValueError('Invalid data format: {}'.format(data_format))
        mean = self.mean.get(in_)
        std = self.std.get(in_)
        pad = self.pad.get(in_)
        pad_value = self.pad_value.get(in_)
        scale = self.scale.get(in_)
        mirror = self.mirror.get(in_)
        center = self.center.get(in_)
        transpose = self.transpose.get(in_)
        in_dims = self.inputs[in_][1:-1]
        if mean is not None:
            data_in -= mean
        if std is not None:
            data_in /= std
        if pad is not None:
            if pad_value is None:
                pad_value = 0
            data_in = np.pad(data_in, ((pad,pad), (pad,pad), (0,0)),
                             'constant', constant_values=pad_value)
        if scale is not None:
            if scale.shape == (2,2):
                # rand scale, individual ratios
                randsh = self.rand.rand()
                randsw = self.rand.rand()
                scaleh = scale[0,0]*(1-randsh) + scale[0,1]*randsh
                scalew = scale[1,0]*(1-randsw) + scale[0,1]*randsh
            elif scale.shape == (2,):
                # rand scale, keep the ratio of h and w
                randsc = self.rand.rand()
                scaleh = scale[0]*(1-randsc) + scale[1]*randsc
                scalew = scale[0]*(1-randsc) + scale[1]*randsc
            elif scale.shape == (2,1):
                # fixed scale
                # If scaleh/w == -1, the input data is scaled to have the same
                # height/width with the input blob.
                scaleh = scale[0]
                scalew = scale[1]
                if scaleh == -1:
                    scaleh = 1.0 * in_dims[0] / data_in.shape[0]
                if scalew == -1:
                    scalew = 1.0 * in_dims[1] / data_in.shape[1]
            elif scale.shape == (1,):
                # fixed scale, keep the ratio of h and w
                # If scale == -1, the input data is scaled to have the same
                # height/width with the input blob and the other edge is not
                # shorter than the corresponding edge of the input blob.
                if scale[0] == -1:
                    larger_scale = np.max([1.0 * in_dims[0] / data_in.shape[0],
                                          1.0 * in_dims[1] / data_in.shape[1]])
                    scaleh = larger_scale
                    scalew = larger_scale
                else:
                    scaleh = scale[0]
                    scalew = scale[0]
            else:
                scaleh = 1.0
                scalew = 1.0
            # bilinear interpolation
            data_in = ndimage.zoom(data_in, (scaleh, scalew, 1.0), order=1)
            
        if data_in.shape[0] >= in_dims[0] and data_in.shape[1] >= in_dims[1]:
            if center is not None and center:
                h_off = int((data_in.shape[0] - in_dims[0] + 1)/2)
                w_off = int((data_in.shape[1] - in_dims[1] + 1)/2)
            else:
                h_off = self.rand.randint(data_in.shape[0] - in_dims[0] + 1)
                w_off = self.rand.randint(data_in.shape[1] - in_dims[1] + 1)
            data_in = data_in[h_off:h_off+in_dims[0], w_off:w_off+in_dims[1], :]
        else:
            raise ValueError('Image is smaller than input: {} vs {}'.format(
                             data_in.shape[:2], in_dims))
        if mirror is not None and mirror and self.rand.randint(2) == 1:
            data_in = data_in[:,::-1,:]
        if transpose is not None:
            data_in = np.transpose(data_in, transpose)
        return data_in
    
    def deprocess(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        std = self.std.get(in_)
        mean = self.mean.get(in_)
        transpose = self.transpose.get(in_)
        if transpose is not None:
            data_in = np.transpose(data_in, np.argsort(transpose))
        if std is not None: 
            data_in *= std
        if mean is not None:
            data_in += mean
        return data_in

def get_plotable_data(data):
    data[data < 0] = 0
    data[data > 255] = 255
    data = data.swapaxes(0,1).swapaxes(1,2)
    data = np.require(data, dtype=np.uint8)
    return data
