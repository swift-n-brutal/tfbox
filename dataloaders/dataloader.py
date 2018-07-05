# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:38 2017

@author: shiwu_001
"""

import numpy as np
from collections import OrderedDict
from multiprocessing import Queue, Process
from copy import deepcopy

from ..config import NP_DTYPE as DTYPE

from .util import Handler
#import signal
#class Handler:
#    def __init__(self, idx=-1):
#        self.alive = True
#        self.idx = idx
#        signal.signal(signal.SIGINT, self.on_interruption)
#        signal.signal(signal.SIGTERM, self.on_termination)
#
#    def on_interruption(self, sig, frame):
#        pass
#        #print self.idx, 'Got SIGINT'
#
#    def on_termination(self, sig, frame):
#        #print self.idx, 'Got SIGTERM'
#        self.alive = False

class DataLoader(object):
    """Base Class of Data Loader
    
    A data loader is an iterator over a dataset. It has
        - a dataset that describes the information about the dataset.
        - a sampler that samples indices out of the size of the dateset.
        - multiple workers that preprocess the data to be loaded.
        - a sampler queue that gets indices from the sampler and feeds
            them to the workers.
        - a data queue that gets preprocessed data from the workers.

    Attributes
    ----------
    dataset : dict
    sampler_queue : mutliprocessing.Queue
    sampler_process : multiprocessing.Process
    data_queue : multiprocessing.Queue
    worker_processes : list of multiprocessing.Process
    """
    def __init__(self, args, phase, queue_size=4, seed=None,
                 transformer=None, name='dataloader'):
        """__init__ method of DataLoader

        Parameters
        ----------
        args : dict
        phase : str
            One of 'train' or 'test'.
        queue_size : int
            The size of queues for the sampler and the workers.
        seed : int or None
            The seed used by the sampler.
        transformer : tfbox.dataloaders.Transformer or None
            The transformer used by the workers or the parent process.
            Note that the seeds of individual workers are set when calling
            add_prefetch_process().
        name : str
        """
        self.name = name
        self.dataset = self._init_dataset(args, phase, transformer)
        self.rand = np.random.RandomState(seed)
        # process to sample indices
        self.sampler_queue_size = queue_size
        self.sampler_queue = None
        self.sampler_process = None
        # processes to load data
        self.data_queue_size = queue_size
        self.data_queue = None
        self.worker_processes = list()
        # NOTE Close normally only if all workers submit
        # their results within the wait time.
        self.__wait_time = 2
    
    @classmethod
    def _init_dataset(cls, *args):
        raise NotImplementedError

    @classmethod
    def _load_data(cls, *args):
        raise NotImplementedError

    @classmethod
    def _sampler_process(cls, rand, n_samples, batch_size, sampler_queue):
        handler = Handler()
        while handler.alive:
            sampler_queue.put(rand.choice(n_samples, size=batch_size, replace=False))
        sampler_queue.close()

    @classmethod
    def _worker_process(cls, dataset, blob_shapes, sampler_queue, data_queue, seed):
        handler = Handler()
        # independent random seed
        dataset['rand'] = np.random.RandomState(seed)
        #
        prefetch_data = OrderedDict()
        for k in blob_shapes.keys():
            prefetch_data[k] = np.zeros(blob_shapes[k], dtype=DTYPE)
        while handler.alive:
            if sampler_queue is None:
                indices = None
            else:
                indices = sampler_queue.get()
            cls._load_data(dataset, prefetch_data, indices)
            data_queue.put((indices, deepcopy(prefetch_data)))
        data_queue.close()

    def _get_sampler_queue(self, batch_size):
        if self.sampler_queue is None:
            if self.dataset['n_samples'] > 0:
                q = Queue(self.sampler_queue_size)
                p = Process(name='/%s/sampler' % self.name, target=self._sampler_process,
                        args=(self.rand, self.dataset['n_samples'], batch_size, q))
                p.start()
                self.sampler_process = p
                self.sampler_queue = q
        else:
            print "Sampler process for %s already exists" % self.name
        return self.sampler_queue

    def add_prefetch_process(self, blob_shapes, nproc=1, seeds=None):
        """
        Parameters
        ----------
        blob_shapes : OrderedDict
            Key-value pairs of (blob_name, blob_shape). Note that all the
            blob_shape's should share the same batch size.
        n_proc : int
            Number of prefetch processes.
        seeds : None or int or list of int
        """
        blob_name = blob_shapes.keys()[0]
        blob_shape = blob_shapes[blob_name]
        sampler_queue = self._get_sampler_queue(blob_shape[0])
        assert self.data_queue is None, 'data queue for %s already exists' % self.name
        self.data_queue = Queue(self.data_queue_size)
        for i in xrange(nproc):
            if type(seeds) is list and len(seeds) >= nproc:
                seed = seeds[i]
            else:
                seed = None
            wp = Process(name='/%s/worker%d' % (self.name, i), target=self._worker_process,
                    args=(self.dataset, blob_shapes, sampler_queue, self.data_queue, seed))
            wp.start()
            self.worker_processes.append(wp)

    def get_data(self, blobs, indices=None):
        assert blobs != None or indices == None, \
                "Blobs array should be provided when indices is specified."
        if indices is None:
            return self.data_queue.get()
        else:
            self._load_data(self.dataset, blobs, indices)
            return indices, blobs

    def clean_and_close(self):
        from Queue import Empty
        # first terminate all worker processes
        for wp in self.worker_processes:
            wp.terminate()
        if self.data_queue is not None:
            try:
                while True:
                    self.data_queue.get(timeout=self.__wait_time)
            except Empty:
                pass
            self.data_queue.close()
            self.data_queue.join_thread()
        for wp in self.worker_processes:
            while wp.is_alive():
                wp.join(timeout=1)
            print "Joined process:", wp.name
        # terminate sampler process
        if self.sampler_process is not None:
            self.sampler_process.terminate()
        if self.sampler_queue is not None:
            try:
                while True:
                    self.sampler_queue.get(timeout=self.__wait_time)
            except Empty:
                pass
            self.sampler_queue.close()
            self.sampler_queue.join_thread()
        if self.sampler_process is not None:
            while self.sampler_process.is_alive():
                self.sampler_process.join(timeout=1)
            print "Joined process:", self.sampler_process.name
    
    def __del__(self):
        self.clean_and_close()
        print "Joined all prefetch processes."
