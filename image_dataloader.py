# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 17:15:03 2017

@author: shiwu_001
"""

import os
import os.path as osp
import numpy as np
from scipy import misc, ndimage
from config import NP_DTYPE as DTYPE

class ImageDataLoader(object):
    def __init__(self, folder, names, transformer=None, seed=None):
        self.rand = np.random.RandomState(seed)
        self.folder = folder
        self.transformer = transformer
        self.names = self._init_name_list(names)
        
    def _init_name_list(self, names):
        fp = file(names, 'r')
        name_list = []
        for line in fp:
            name_list.append(line.strip('\n'))
        self.n_images = len(name_list)
        print self.n_images, 'images in total'
        return name_list
        
    def _load_batch(self, batchids, blob_name, dest):
        for i, index in enumerate(batchids):
            im_path = osp.join(self.folder, self.names[index])
            im = misc.imread(im_path)
#            im = im.swapaxes(0,2).swapaxes(1,2)
            if self.transformer is not None:
                im = self.transformer.process(blob_name, im)
            else:
                im = im.astype(DTYPE)
            assert dest[i,...].shape == im.shape, \
                'blob shape is not equal to image shape: {} vs {}'.format(dest[i,...].shape, im.shape)
            dest[i,...] = im[...]
    
    def sample_batch(self, batchsize):
        return self.rand.choice(self.n_images, size=batchsize, replace=False)
            
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blobs)
        assert len(blob_names) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blob_names)
        blob = blobs[0]
        blob_name = blob_names[0]
        if batchids is None:
            batchids = self.sample_batch(blob.shape[0])
        self._load_batch(batchids, blob_name, blob.data)
        return batchids

from multiprocessing import Queue, Process
class ImageDataLoaderPrefetch(ImageDataLoader):
    def __init__(self, queue_size, folder, names, transformer=None, seed=None):
        super(ImageDataLoaderPrefetch, self).__init__(
            folder, names, transformer=transformer, seed=seed)
        # process to sample batchid
        self.batchids_queue_size = queue_size
        self.batchids_queue = None
        self.batchids_process = None
        # processes to load data
        self.data_queue_size = queue_size
        self.blob_names = list()
        self.data_queue_ = dict()
        self.data_shape_ = dict()
        self.worker_processes = list()

    def add_prefetch_process(self, blob_name, data_shape, nproc=1, seeds=None):
        batchsize = data_shape[0]
        if self.batchids_process is None:
            self._init_batchids_process(batchsize)
        self.blob_names.append(blob_name)
        self.data_shape_[blob_name] = data_shape
        data_queue = Queue(self.data_queue_size)
        self.data_queue_[blob_name] = data_queue
        for i in xrange(nproc):
            if type(seeds) is list:
                seed = seeds[i]
            else:
                seed = None
            wp = Process(target=self.__class__._worker_process,
                         args=(blob_name, data_shape, data_queue, self.batchids_queue,
                               self.folder, self.names, self.transformer, seed))
            wp.start()
            self.worker_processes.append(wp)
    
    def _init_batchids_process(self, batchsize):
        if self.batchids_process is not None:
            print 'Batchids process already exists'
            return
        self.batchids_queue = Queue(self.batchids_queue_size)
        self.batchids_process = Process(target=self._batchids_process,
                                        args=(self.rand, self.n_images, batchsize, self.batchids_queue))
        self.batchids_process.start()
    
    @classmethod
    def _batchids_process(cls, rand, n_images, batchsize, batchids_queue):
        while True:
            batchids_queue.put(rand.choice(n_images, size=batchsize, replace=False))
        
    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
                        folder, names, transformer, seed):
        # independent seed
        transformer.rand = np.random.RandomState(seed)
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        while True:
            batchids = batchids_queue.get()
            for i, index in enumerate(batchids):
                im_path = osp.join(folder, names[index])
                im = misc.imread(im_path, mode='RGB')
#                im = im.swapaxes(0,2).swapaxes(1,2)
                if transformer is not None:
                    im = transformer.process(blob_name, im)
                else:
                    im = im.astype(DTYPE)
                assert prefetch_data[i,...].shape == im.shape, 'blob shape is not equal to image shape'
                prefetch_data[i,...] = im[...]
            data_queue.put((batchids, prefetch_data.copy()))
        
    def _get_data(self, blob_name):
        """
            Return a batch of data for the blob named by blob_name
        """
        dq = self.data_queue_.get(blob_name)
        assert dq is not None, 'No such blob specified: %s' % blob_name
        return dq.get()
        
    def clean_and_close(self):
        if self.batchids_process is not None:
            self.batchids_process.terminate()
            self.batchids_process.join()
        for wp in self.worker_processes:
            wp.terminate()
            wp.join()
        if self.batchids_queue is not None:
            self.batchids_queue.close()
            self.batchids_queue.join_thread()
        for name in self.blob_names:
            dq = self.data_queue_[name]
            dq.close()
            dq.join_thread()

    def __del__(self):
        self.clean_and_close()
        print 'Closed prefetch processes.'
    
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blobs)
        assert len(blob_names) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blob_names)
        blob = blobs[0]
        blob_name = blob_names[0]
        if batchids is None:
            batchids, prefetch_data = self._get_data(blob_name)
            blob[...] = prefetch_data
        else:
            self.dataloader._load_batch(batchids, blob_name, blob)
        return batchids

class ImageTransformer(object):
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
        elif scale.shape != (2, 1) and scale.shape != (2,) and scale.shape != (2,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale[in_] = scale

    def set_scale2(self, in_, scale2):
        self.__check_input(in_)
        if scale2.shape == () or scale2.shape == (1,):
            scale2 = scale2.reshape(1)
        elif scale2.shape != (2, 1) and scale2.shape != (2,) and scale2.shape != (2,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale2[in_] = scale2
            
    def set_mean(self, in_, mean):
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][3]:
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
            if ss[0] != self.inputs[in_][3]:
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
        in_dims = self.inputs[in_][1:3]
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
            if scale2.shape == (2,2):
                # rand scale, individual ratios
                randsh = self.rand.rand()
                randsw = self.rand.rand()
                scale2h = scale2[0,0]*(1-randsh) + scale2[0,1]*randsh
                scale2w = scale2[1,0]*(1-randsw) + scale2[0,1]*randsh
            elif scale2.shape == (2,):
                # rand scale, keep the ratio of h and w
                randsc = self.rand.rand()
                scale2h = scale2[0]*(1-randsc) + scale2[1]*randsc
                scale2w = scale2[0]*(1-randsc) + scale2[1]*randsc
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
                    scale2h = scale2[0]
                    scale2w = scale2[0]
            else:
                scale2h = data_in.shape[0]
                scale2w = data_in.shape[1]
            # bilinear interpolation
            scaleh = scale2h / data_in.shape[0]
            scalew = scale2w / data_in.shape[1]
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
                             data_in.shape[0:2], in_dims))
        if mirror is not None and mirror and self.rand.randint(2) == 1:
            data_in = data_in[:,::-1,:]
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


import xml.etree.ElementTree as et
class BBoxImageDataLoaderPrefetch(ImageDataLoaderPrefetch):
    def __init__(self, bbox, bbox_folder, qsize, img_folder, names,
                 transformer=None, seed=None):
        super(BBoxImageDataLoaderPrefetch, self).__init__(
            qsize, img_folder, names, transformer=transformer, seed=seed)
        self.bbox_folder = bbox_folder
        self.bbox = bbox
        if bbox is not None and bbox_folder is not None:
            print 'WARNING: bbox ans bbox_folder can not be set at the same time.'
            print 'Use bbox as default.'
            self.bbox_folder = None
    
    def _init_bbox(self):
        """
            deprecated
        """
        bbox_folder = self.bbox_folder
        bbox = np.zeros((self.n_images, 4), dtype=DTYPE)
        for i, name in enumerate(self.names):
            bbpath = osp.join(bbox_folder, name[:-5] + '.xml')
            tree = et.parse(bbpath)
            node = tree.find('object/bndbox')
            xmin = np.float32(node.findtext('xmin')) - 1
            xmax = np.float32(node.findtext('xmax')) - 1
            ymin = np.float32(node.findtext('ymin')) - 1
            ymax = np.float32(node.findtext('ymax')) - 1
            anno_size = tree.find('size')
            anno_w = np.float32(anno_size.findtext('width'))
            anno_h = np.float32(anno_size.findtext('height'))
            bbox[i,:] = [xmin/anno_w, xmax/anno_w, ymin/anno_h, ymax/anno_h]
        return bbox

    def add_prefetch_process(self, blob_name, data_shape):
        batchsize = data_shape[0]
        if self.batchids_process is None:
            self._init_batchids_process(batchsize)
        self.blob_names.append(blob_name)
        self.data_shapes[blob_name] = data_shape
        data_queue = Queue(self.data_queue_size)
        self.data_queues[blob_name] = data_queue
        wp = Process(target=BBoxImageDataLoaderPrefetch._worker_process,
                     args=(blob_name, data_shape, data_queue, self.batchids_queue,
                           self.folder, self.names, self.transformer,
                           self.bbox, self.bbox_folder))
        wp.start()
        self.worker_processes.append(wp)
    
    @classmethod
    def _get_bbox(cls, bbox_folder, name):
        bbox = np.zeros(4, dtype=np.float32)
        bbox_path = osp.join(bbox_folder, name[:-5] + '.xml')
        tree = et.parse(bbox_path)
        # extract the position of the bounding box
        node = tree.find('object/bndbox')
        xmin = np.float32(node.findtext('xmin'))
        xmax = np.float32(node.findtext('xmax'))
        ymin = np.float32(node.findtext('ymin'))
        ymax = np.float32(node.findtext('ymax'))
        # extract the size of image showed to the annotator
        anno_size = tree.find('size')
        anno_w = np.float32(anno_size.findtext('width'))
        anno_h = np.float32(anno_size.findtext('height'))
        bbox[...] = [xmin/anno_w, xmax/anno_w, ymin/anno_h, ymax/anno_h]
        return bbox
        
    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
                        folder, names, transformer, bbox, bbox_folder):
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        while True:
            batchids = batchids_queue.get()
            for i, index in enumerate(batchids):
                name = names[index]
                im_path = osp.join(folder, name)
                im = misc.imread(im_path, mode='RGB')
                im = im.swapaxes(0,2).swapaxes(1,2)
                # crop roi
                if bbox_folder is not None:
                    bbox = cls._get_bbox(bbox_folder, name)
                xmin = int(bbox[0] * im.shape[2])
                xmax = int(bbox[1] * im.shape[2])
                ymin = int(bbox[2] * im.shape[1])
                ymax = int(bbox[3] * im.shape[1])
                im = im[:,ymin:ymax,xmin:xmax]
                if transformer is not None:
                    im = transformer.process(blob_name, im)
                else:
                    im = im.astype(DTYPE)
                assert prefetch_data[i,...].shape == im.shape, 'blob shape is not equal to image shape'
                prefetch_data[i,...] = im[...]
            data_queue.put(prefetch_data)

class CelebADataLoaderPrefetch(ImageDataLoaderPrefetch):
    ROI_WIDTH=128
    ROI_VALUE=255
    OROI_VALUE=0
    def __init__(self, queue_size, folder, names, transformer=None, seed=None):
        super(CelebADataLoaderPrefetch, self).__init__(
            queue_size, folder, names, transformer=transformer, seed=seed)
    
    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
                        folder, names, transformer):
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        while True:
            batchids = batchids_queue.get()
#            self.dataloader._load_batch(batchids, blob_name, prefetch_data)
            for i, index in enumerate(batchids):
                im_path = osp.join(folder, names[index])
                im = misc.imread(im_path, mode='RGB')
                im = im.swapaxes(0,2).swapaxes(1,2)
                # add alpha channel
                c,h,w = im.shape
                im = np.resize(im, (c+1, h, w))
                im[3,...] = cls.OROI_VALUE
                h_off = (h-cls.ROI_WIDTH) / 2
                w_off = (w-cls.ROI_WIDTH) / 2
                im[3, h_off:h_off+cls.ROI_WIDTH, w_off:w_off+cls.ROI_WIDTH] = cls.ROI_VALUE
                if transformer is not None:
                    im = transformer.process(blob_name, im)
                else:
                    im = im.astype(DTYPE)
                assert prefetch_data[i,...].shape == im.shape, 'blob shape is not equal to image shape'
                prefetch_data[i,...] = im[...]
            data_queue.put(prefetch_data)
            
def get_plotable_data(data):
    data[data < 0] = 0
    data[data > 255] = 255
    data = data.swapaxes(0,1).swapaxes(1,2)
    data = np.require(data, dtype=np.uint8)
    return data

import matplotlib.pyplot as plt
if __name__ == '__main__':
    # imagenet cat
#    bbox = None
#    bbox_folder = 'e:/images/ILSVRC2012_bbox_train_v2'
#    img_folder = 'e:/images/imagenet/train'
#    img_names = 'e:/projects/cpp/caffe-windows-ms/examples/latte_v2/test/cat_loc.txt'
    # celeba
    bbox = np.array([0.0/178, 178.0/178, 0.0/218, 218.0/218])
    bbox_folder = None
    img_folder = 'e:/images/img_align_celeba'
    img_names = 'e:/images/img_align_celeba/names.txt'
    blob_name = 'input'
    blob_shape = np.array([10, 3, 64, 64])
    inputs = {blob_name: blob_shape}
    scale = np.array([64.0 / (45 + 128)])
    tf = ImageTransformer(inputs)
    tf.set_scale(blob_name, scale)
    tf.set_center(blob_name, False)
    bboxloader = BBoxImageDataLoaderPrefetch(bbox, bbox_folder, 2, img_folder,
                                             img_names, transformer=tf)
    try:
        bboxloader.add_prefetch_process(blob_name, inputs[blob_name])
        for i in xrange(2):
            data = bboxloader._get_data(blob_name)
            n_samples = data.shape[0]
            for j in xrange(n_samples):
                im = tf.deprocess(blob_name, data[j,...])
                im = get_plotable_data(im)
                print i, j, im.shape
                plt.imshow(im)
                plt.waitforbuttonpress()
                plt.close()
    finally:
        bboxloader.clean_and_close()
        print 'Close all processes. Program exits.'
    
