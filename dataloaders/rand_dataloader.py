# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 16:38:38 2017

@author: shiwu_001
"""

import numpy as np

from .util import Handler
from ..config import NP_DTYPE as DTYPE

class RandDataLoader(object):
    def __init__(self, rand_type="gaussian", std=1.0, seed=None):
#        super(RandDataLoader, self).__init__(seed=seed)
        self.rand = np.random.RandomState(seed)
        self.rand_type = rand_type
        self.std = std
    
    def fill_input(self, blobs, blob_names, batchids):
        if self.rand_type == "gaussian":
            for b in blobs:
                b[...] = self.std * self.rand.randn(b.count).reshape(b.shape)
        elif self.rand_type == "uniform":
            a = np.sqrt(3) * self.std
            for b in blobs:
                b[...] = self.rand.uniform(-a, a, size=b.shape)
        else:
            raise ValueError("Not supported rand_type: %s" %
                             self.rand_type)


from multiprocessing import Queue, Process
class RandDataLoaderPrefetch(RandDataLoader):
    def __init__(self, queue_size, seed=None):
        self.rand = np.random.RandomState(seed)
        # processes to sample data
        self.data_queue_size = queue_size
        self.blob_names = list()
        self.data_queue_ = dict()
        self.data_shape_ = dict()
        self.worker_processes = list()

    def add_prefetch_process(self, blob_name, data_shape, nproc=1,
            rand_type="gaussian", mean=0., std=1., seeds=None):
        self.blob_names.append(blob_name)
        self.data_shape_[blob_name] = data_shape
        data_queue = Queue(self.data_queue_size)
        self.data_queue_[blob_name] = data_queue
        for i in xrange(nproc):
            if seeds is None:
                seed = None
            else:
                assert type(seeds) == list and len(seeds) == nproc, \
                        "Invalid random seeds"
                seed = seeds[i]
            wp = Process(target=self.__class__._worker_process,
                    args=(blob_name, data_shape, data_queue, 
                        rand_type, mean, std, seed))
            wp.start()
            self.worker_processes.append(wp)

    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue,
            rand_type, mean, std, seed):
        rand = np.random.RandomState(seed)
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        handler = Handler()
        if rand_type == "gaussian":
            while handler.alive:
                prefetch_data[...] = mean + std*rand.standard_normal(data_shape)
                data_queue.put(prefetch_data.copy())
        elif rand_type == "uniform":
            a = np.sqrt(3) * std
            while handler.alive:
                prefetch_data[...] = mean + a*rand.uniform(-a, a, size=data_shape)
                data_queue.put(prefetch_data.copy())
        else:
            raise ValueError("Not supported rand_type (%s)" % rand_type)
        data_queue.close()

    def _get_data(self, blob_name):
        dq = self.data_queue_.get(blob_name)
        assert dq is not None, "No such blob (%s)" % blob_name
        return dq.get()

    def clean_and_close(self):
        from Queue import Empty
        for wp in self.worker_processes:
            wp.terminate()
        for name in self.blob_names:
            dq = self.data_queue_[name]
            try:
                while True:
                    dq.get(timeout=1)
            except Empty:
                pass
            dq.close()
            dq.join_thread()
            print "Closed data_queue:", name
        for wp in self.worker_processes:
            wp.join()
            print "Joined process:", wp.name

    def __del__(self):
        self.clean_and_close()
        print 'Closed prefetch processes.'

    def fill_input(self, blobs, blob_names):
        assert len(blobs) == 1, "RandDataLoader fills one input blob (%d given)" % (len(blobs))
        blobs[0][...] = self._get_data(blob_names[0])
        return 0

def main():
    rdl = RandDataLoaderPrefetch(4)
    input_name = 'rand'
    input_shape = [2, 3]
    rand_vec = np.zeros(input_shape, dtype=DTYPE)
    rdl.add_prefetch_process(input_name, rand_vec.shape, 2, mean=10, seeds=[37, 137])
    print rand_vec
    rdl.fill_input([rand_vec], [input_name])
    print rand_vec
    print rdl._get_data(input_name) 
    rdl.fill_input([rand_vec], [input_name])
    print rand_vec

if __name__ == '__main__':
    main()
