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

class ImagenetDataLoader(object):
    def __init__(self, folder, names, transformer=None, seed=None):
        self.rand = np.random.RandomState(seed)
        self.folder = folder
        self.transformer = transformer
        self.names, self.labels = self._init_name_list(names)
        
    def _init_name_list(self, names):
        fp = file(names, 'r')
        name_list = list()
        label_list = list()
        for line in fp:
            line = line.strip('\n')
            name, label = line.split(' ')
            name_list.append(name)
            label_list.append(int(label))
        self.n_images = len(name_list)
        print self.n_images, 'images in total'
        return name_list, label_list
        
    def _load_batch(self, blobs, blob_names, batchids):
        has_label = len(blobs) == 2
        blob_name = blob_names[0]
        for i, index in enumerate(batchids):
            im_path = osp.join(self.folder, self.names[index])
            im = misc.imread(im_path, mode='RGB')
#            im = im.swapaxes(0,2).swapaxes(1,2)
            if self.transformer is not None:
                im = self.transformer.process(blob_name, im)
            else:
                im = im.astype(DTYPE)
            assert blobs[0].shape[1:] == im.shape, \
                    'blob shape is not equal to image shape: {} vs {}'.format(blobs[0].shape[1:], im.shape)
            blobs[0][i,...] = im[...]
            if has_label:
                blobs[1][i] = self.labels[index]
    
    def sample_batch(self, batchsize):
        return self.rand.choice(self.n_images, size=batchsize, replace=False)
            
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blobs)
        assert len(blob_names) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blob_names)
        blob = blobs[0]
        blob_name = blob_names[0]
        if batchids is None:
            batchids = self.sample_batch(blob.shape[0])
        self._load_batch(blobs, blob_names, batchids)
        return batchids

from multiprocessing import Queue, Process
class ImagenetDataLoaderPrefetch(ImagenetDataLoader):
    def __init__(self, folder, names, queue_size=4, transformer=None, seed=None):
        super(ImagenetDataLoaderPrefetch, self).__init__(
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
                               self.folder, self.names, self.labels, self.transformer, seed))
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
                        folder, names, labels, transformer, seed):
        # independent ssed
        transformer.rand = np.random.RandomState(seed)
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        prefetch_labels = np.zeros(data_shape[0], dtype=DTYPE)
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
                assert prefetch_data.shape[1:] == im.shape, 'blob shape is not equal to image shape'
                prefetch_data[i,...] = im[...]
                prefetch_labels[i] = DTYPE(labels[index])
            data_queue.put((batchids, (prefetch_data.copy(), prefetch_labels.copy())))
        
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
        assert len(blobs) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blobs)
        assert len(blob_names) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blob_names)
        blob_name = blob_names[0]
        if batchids is None:
            batchids, (images, labels) = self._get_data(blob_name)
            blobs[0][...] = images
            if len(blobs) == 2:
                blobs[1][...] = labels
        else:
            self._load_batch(blobs, blob_names, batchids)
        return batchids
    
class ImagenetDataLoaderMemory(ImagenetDataLoaderPrefetch):
    def __init__(self, folder, names, queue_size=4, transformer=None, seed=None):
        super(ImagenetDataLoaderMemory, self).__init__(
                folder, names, queue_size, transformer, seed)

    def load_all(self, blob_name, blob_shape, nproc=2):
        batchsize = blob_shape[0]
        self._init_batchids_process(batchsize)
        self.add_prefetch_process(blob_name, blob_shape, nproc)
        # change n to the number of images
        n_images = self.n_images
        blob_shape[0] = n_images
        self.data = np.zeros(blob_shape, dtype=DTYPE)
        count = 0
        print 'Loaded',
        for _ in xrange(0, n_images, batchsize):
            batchids, (data, labels) = self._get_data(blob_name)
            for i, idx in enumerate(batchids):
                self.data[idx,...] = data[i,...]
            count += 1
            if count % 10 == 0:
                print count,
        print
    
    def _init_batchids_process(self, batchsize):
        if self.batchids_process is not None:
            print 'batchids process already exists'
            return
        self.batchids_queue = Queue(self.batchids_queue_size)
        self.batchids_process = Process(target=self.__class__._batchids_process,
                args=(self.rand, self.n_images, batchsize, self.batchids_queue))
        self.batchids_process.start()

    @classmethod
    def _batchids_process(cls, rand, n_images, batchsize, batchids_queue):
        batchids = np.zeros(batchsize, dtype=np.int32)
        for start_id in xrange(0, n_images, batchsize):
            stop_id = min(start_id + batchsize, n_images)
            batchids[:stop_id - start_id] = np.arange(start_id, stop_id)
            batchids_queue.put(batchids.copy())
    
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blobs)
        assert len(blob_names) in [1, 2], 'ImageDataLoader fills 1 or 2 input blobs (%d given)' % len(blob_names)
        blob_name = blob_names[0]
        if batchids is None:
            batchids, (images, labels) = self._get_data(blob_name)
            blobs[0][...] = images
            if len(blobs) == 2:
                blobs[1][...] = labels
        else:
            #self._load_batch(blobs, blob_names, batchids)
            for i, idx in enumerate(batchids):
                blobs[0][i,...] = self.data[idx,...]
                blobs[1][i] = self.labels[idx]
        return batchids

