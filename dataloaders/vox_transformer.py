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

class VoxTransformer(object):
    def __init__(self, inputs, seed=None):
        assert type(inputs) == dict, 'Tranformer.inputs = {blob_name: blob_shape}'
        self.rand = np.random.RandomState(seed)
        self.inputs = inputs
        self.scale = dict()
        self.scale2 = dict()
        self.pad = dict()
        self.pad_value = dict()
        self.mean = dict()
        self.std = dict()
        self.mirror = dict()
        self.center = dict()

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception("{} is not one of the net inputs: {}".format(
                            in_, self.inputs))
            
    def set_scale(self, in_, scale):
        self.__check_input(in_)
        if scale.shape == () or scale.shape == (1,):
            scale = scale.reshape(1)
        elif scale.shape != (3,1) and scale.shape != (2,) and scale.shape != (3,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale[in_] = scale

    def set_scale2(self, in_, scale2):
        self.__check_input(in_)
        if scale2.shape == () or scale2.shape == (1,):
            scale2 = scale2.reshape(1)
        elif scale2.shape != (3,1) and scale2.shape != (2,) and scale2.shape != (3,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale2[in_] = scale2
            
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
            if len(ms) == 3:
                ms = ms + (1,)
            if len(ms) != 4:
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
            if len(ss) == 3:
                ss = ss + (1,)
            if len(ss) != 4:
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

    def process(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        mean = self.mean.get(in_)
        std = self.std.get(in_)
        pad = self.pad.get(in_)
        pad_value = self.pad_value.get(in_)
        scale = self.scale.get(in_)
        scale2 = self.scale2.get(in_)
        mirror = self.mirror.get(in_)
        center = self.center.get(in_)
        in_dims = self.inputs[in_][1:-1]
        if mean is not None:
            data_in -= mean
        if std is not None:
            data_in /= std
        if pad is not None:
            if pad_value is None:
                data_in = np.pad(data_in, ((pad,pad), (pad,pad), (pad,pad), (0,0)),
                        'edge')
            else:
                data_in = np.pad(data_in, ((pad,pad), (pad,pad), (pad,pad), (0,0)),
                        'constant', constant_values=pad_value)
        if scale is not None:
            raise ValueError("Not implemented: scale")
            assert scale2 is None, "Cannot set 'scale' and 'scale2' at the same time."
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
        elif scale2 is not None:
            raise ValueError("Not implemented: scale2")
            if scale2.shape == (2,2):
                # rand scale, individual ratios
                randsh = self.rand.rand()
                randsw = self.rand.rand()
                scale2h = scale2[0,0]*(1-randsh) + scale2[0,1]*randsh
                scale2w = scale2[1,0]*(1-randsw) + scale2[0,1]*randsh
            elif scale2.shape == (2,):
                # rand scale, keep the ratio of h and w with the shorter edge to be the random scale
                randsc = self.rand.rand()
                scale2shorter = scale2[0]*(1-randsc) + scale2[1]*randsc
                if data_in.shape[0] > data_in.shape[1]:
                    scale2h = scale2shorter * data_in.shape[0] / data_in.shape[1]
                    scale2w = scale2shorter
                else:
                    scale2h = scale2shorter
                    scale2w = scale2shorter * data_in.shape[1] / data_in.shape[0]
            elif scale2.shape == (2,1):
                # fixed scale
                # If scaleh/w == -1, the input data is scaled to have the same
                # height/width with the input blob.
                scale2h = scale2[0]
                scale2w = scale2[1]
                if scale2h == -1:
                    scale2h = 1.0 * in_dims[0]
                if scale2w == -1:
                    scale2w = 1.0 * in_dims[1]
            elif scale2.shape == (1,):
                # fixed scale, keep the ratio of h and w
                # If scale == -1, the input data is scaled to have the same
                # height/width with the input blob and the other edge is not
                # shorter than the corresponding edge of the input blob.
                if scale2[0] == -1:
                    larger_scale = np.max([1.0 * in_dims[0] / data_in.shape[0],
                                          1.0 * in_dims[1] / data_in.shape[1]])
                    scale2h = larger_scale * data_in.shape[0]
                    scale2w = larger_scale * data_in.shape[1]
                else:
                    scale2shorter = DTYPE(scale2[0])
                    if data_in.shape[0] > data_in.shape[1]:
                        scale2h = scale2shorter * data_in.shape[0] / data_in.shape[1]
                        scale2w = scale2shorter
                    else:
                        scale2h = scale2shorter
                        scale2w = scale2shorter * data_in.shape[1] / data_in.shape[0]
            else:
                scale2h = data_in.shape[0]
                scale2w = data_in.shape[1]
            # bilinear interpolation
            scaleh = scale2h / data_in.shape[0]
            scalew = scale2w / data_in.shape[1]
            data_in = ndimage.zoom(data_in, (scaleh, scalew, 1.0), order=1)
            
        if data_in.shape[0] >= in_dims[0] and data_in.shape[1] >= in_dims[1] and data_in.shape[2] >= in_dims[2]:
            if center is not None and center:
                h_off = int((data_in.shape[0] - in_dims[0] + 1)/2)
                w_off = int((data_in.shape[1] - in_dims[1] + 1)/2)
                d_off - int((data_in.shape[2] - in_dims[2] + 1)/2)
            else:
                h_off = self.rand.randint(data_in.shape[0] - in_dims[0] + 1)
                w_off = self.rand.randint(data_in.shape[1] - in_dims[1] + 1)
                d_off = self.rand.randint(data_in.shape[2] - in_dims[2] + 1)
            data_in = data_in[h_off:h_off+in_dims[0], w_off:w_off+in_dims[1], d_off:d_off+in_dims[2], :]
        else:
            raise ValueError('Image is smaller than input: {} vs {}'.format(
                             data_in.shape[0:-1], in_dims))
        if mirror is not None and mirror:
            # z is the upright direction
            # mirror along the x axis
            if self.rand.randint(2) == 1:
                data_in = data_in[::-1,:,:,:]
            # mirror along the y axis
            if self.rand.randint(2) == 1:
                data_in = data_in[:,::-1,:,:]
        return data_in
    
    def deprocess(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        std = self.std.get(in_)
        mean = self.mean.get(in_)
        if std is not None:
            data_in *= std
        if mean is not None:
            data_in += mean
        return data_in
