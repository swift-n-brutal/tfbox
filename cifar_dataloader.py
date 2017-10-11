# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:38 2017

@author: shiwu_001
"""

import numpy as np
import lmdb
import caffe.proto as caproto
from multiprocessing import Queue, Process

#from image_dataloader import ImageTransformer
from tfbox.config import NP_DTYPE as DTYPE

class CifarDataLoader(object):
    def __init__(self, path, phase, queue_size=4, seed=None, key_length=5,
                 transformer=None, mean=None, std=None):
        self.path = path
        self.txn = self._init_db(path)
        self.phase = phase
        self.queue_size = queue_size
        self.rand = np.random.RandomState(seed)
        self.key_length = key_length
        self.mean = mean if mean else np.array([125.3, 123.0, 113.9])
        self.std = std if std else np.array([63.0, 62.1, 66.7])
        self.transformer = transformer
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
        
    def _init_db(self, path):
        env = lmdb.open(path, readonly=True)
        self.n_images = env.stat()['entries']
        print "%d images in total from %s" % (self.n_images, path)
        txn = env.begin()
        return txn
    
    def set_transformer(self, transformer):
        self.transformer = transformer
    
    def add_prefetch_process(self, blob_name, data_shape, nproc=1):
        batchsize = data_shape[0]
        if self.batchids_process is None:
            self._init_batchids_process(batchsize)
        self.blob_names.append(blob_name)
        self.data_shape_[blob_name] = data_shape
        data_queue = Queue(self.data_queue_size)
        self.data_queue_[blob_name] = data_queue
        for i in xrange(nproc):
            wp = Process(target=self.__class__._worker_process,
                         args=(blob_name, data_shape, data_queue, self.batchids_queue,
                               self.transformer, self.path, self.key_length))
            wp.start()
            self.worker_processes.append(wp)
            
    def _init_batchids_process(self, batchsize):
        if self.batchids_process is not None:
            print 'Batchids process already exists.'
            return
        self.batchids_queue = Queue(self.batchids_queue_size)
        self.batchids_process = Process(target=self.__class__._batchids_process,
                                        args=(self.rand, self.n_images, batchsize, self.batchids_queue))
        self.batchids_process.start()
    
    @classmethod
    def _batchids_process(cls, rand, n_images, batchsize, batchids_queue):
        while True:
            batchids_queue.put(rand.choice(n_images, size=batchsize, replace=False))
        
    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
                        transformer, path, key_length):
        # independent random seed
        transformer.rand = np.random.RandomState()
        prefetch_images = np.zeros(data_shape, dtype=DTYPE)
        prefetch_labels = np.zeros(data_shape[0], dtype=DTYPE)
        env = lmdb.open(path, readonly=True)
        txn = env.begin()
        while True:
            batchids = batchids_queue.get()
            for i, index in enumerate(batchids):
                raw_datum = txn.get(eval("'%%0%dd' %% index" % key_length))
                datum = caproto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                x = flat_x.reshape(datum.channels, datum.height, datum.width)
                y = datum.label
                if transformer is not None:
                    x = transformer.process(blob_name, x)
                else:
                    x = x.astype(DTYPE)
                assert prefetch_images[i,...].shape == x.shape, \
                        'blob shape is not consistent with image shape'
                prefetch_images[i,...] = x[...]
                prefetch_labels[i] = DTYPE(y)
            data_queue.put((batchids, (prefetch_images.copy(), prefetch_labels.copy())))
            
    def _get_data(self, blob_name):
        dq = self.data_queue_.get(blob_name)
        assert dq is not None, 'No such blob specified: %s' % blob_name
        return dq.get()
        
    def _load_batch(self, blobs, blob_names, batchids):
        txn = self.txn
        key_length = self.key_length
        transformer = self.transformer
        blob_name = blob_names[0]
        has_label = len(blob_names) == 2
        for i, index in enumerate(batchids):
            raw_datum = txn.get(eval("'%%0%dd' %% index" % key_length))
            datum = caproto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            if transformer is not None:
                x = transformer.process(blob_name, x)
            else:
                x = x.astype(DTYPE)
            assert blobs[0].shape[1:] == x.shape, \
                    'blob shape is not consistent with image shape'
            blobs[0][i,...] = x[...]
            if has_label:
                blobs[1][i] = DTYPE(y)
    
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
        assert len(blobs) in [1,2], 'CifarDataLoader fills one or two input blobs: %d given' % (len(blobs))
        assert len(blob_names) in [1,2], 'CifarDataLoader fills one or two input blobs: %d given' % (len(blob_names))
        if batchids is None:
            batchids, (images, labels) = self._get_data(blob_names[0])
            blobs[0][...] = images
            if len(blobs) == 2:
                blobs[1][...] = labels
        else:
            self._load_batch(blobs, blob_names, batchids)
            
def get_plottable_data(data, data_format="CHW"):
    data = np.clip(np.round(data), 0, 255).astype(np.uint8)
    if data_format == "CHW":
        data = data.swapaxes(0,1).swapaxes(1,2)
    return data
    

def main():
    import matplotlib.pyplot as plt
    print "Test"
    queue_size = 4
    nproc = 2
    train_path = "E:/projects/cpp/caffe-windows-ms/examples/cifar10/cifar10_train_lmdb"
    test_path = "E:/projects/cpp/caffe-windows-ms/examples/cifar10/cifar10_test_lmdb"
    name_data = 'data'
    name_labels = 'labels'
    batchsize = 64
    chn = 3
    data_size = 32
    blob_data = np.zeros([batchsize, chn, data_size, data_size], dtype=DTYPE)
    blob_labels = np.zeros([batchsize], dtype=DTYPE)
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])
    pad = 4
    # train transformer
    tf_train = ImageTransformer({name_data: blob_data.shape})
    tf_train.set_mean(name_data, mean)
    tf_train.set_std(name_data, std)
    tf_train.set_pad(name_data, pad)
    tf_train.set_center(name_data, False)
    tf_train.set_mirror(name_data, True)
    cdl_train = CifarDataLoader(train_path, 'train', queue_size, transformer=tf_train)
    cdl_train.add_prefetch_process(name_data, blob_data.shape, nproc)
    #
    plt.figure(1)
    for i in xrange(10):
        cdl_train.fill_input([blob_data, blob_labels],
                             [name_data, name_labels], batchids=None)
        im = tf_train.deprocess(name_data, blob_data[0,...])
        im = get_plottable_data(im)
        print blob_labels[0]
        plt.imshow(im)
        plt.waitforbuttonpress()
        plt.close()
    
if __name__ == "__main__":
    main()
