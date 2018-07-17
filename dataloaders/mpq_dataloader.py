# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:38 2017

@author: shiwu_001
"""

import numpy as np
from multiprocessing import Process
from multiprocessing import Queue

from argparse import ArgumentParser

from .utils import Handler

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
    name : str
    dataset : tfbox.dataloader.Dataset
    """
    def __init__(self, args, dataset, batch_shape, phase='train', seed=None,
            name='dataloader'):
        """__init__ method of DataLoader

        Parameters
        ----------
        args : dict
            timeout : int or float
                If all workers submit their prefetched data within the waiting
                time, the dataloader will be closed normally. 
            queue_size : int
                The size of queues for the sampler and the workers.
        dataset : Dataset
        batch_shape : tuple of int
            The shape of the batch indices. The fetched data are of shape
            batch_shape + data_shape. 
        phase : str
            One of 'train' or 'test'.
        seed : int or None
            The seed used by the sampler.
        name : str
        """
        self.name = name
        self.dataset = dataset
        self._batch_shape = batch_shape if isinstance(batch_shape, tuple) \
                else tuple(batch_shape)
        self._phase = phase #: Defines the sampling strategy
        self._rand = np.random.RandomState(seed)
        self._timeout = args['dl_timeout']

        # process to sample indices
        self._indices_queue_size = args['dl_indices_queue_size']
        #: multiprocessing.Queue
        self._indices_queue = None
        #: multiprocessing.Process
        self._sampler_process = None

        # processes to load data
        self._data_queue_size = args['dl_data_queue_size']
        #: multiprocessing.Queue
        self._data_queue = None
        #: list of multiprocessing.Process
        self._worker_processes = list()

        self._prefetch_process_added = False

    def add_prefetch_process(self, n_proc=1, seeds=None):
        """
        Parameters
        ----------
        n_proc : int
            The number of prefetch processes.
        seeds : None or list of int
            The seeds used by the datasets respectively.
        """
        assert n_proc >= 1, 'Need at least 1 process to add: %d given' % n_proc
        indices_queue = self._get_indices_queue()
        data_queue = self._get_data_queue()
        if type(seeds) is list or type(seeds) is tuple:
            assert len(seeds) >= n_proc
        else:
            seeds = [None]*n_proc
        for i in xrange(n_proc):
            wp = Process(name='/%s/worker%d' % (self.name, i), target=self._worker,
                    args=(self.dataset, indices_queue, data_queue, self._batch_shape, seeds[i]))
            wp.start()
            self._worker_processes.append(wp)
        self._prefetch_process_added = True

    def _get_indices_queue(self):
        if self._indices_queue is None:
            if self.dataset.size() > 0:
                # Otherwise, the dataset does not need indices.
                q = Queue(self._indices_queue_size)
                p = Process(name='/%s/sampler' % self.name, target=self._sampler,
                        args=(q, self.dataset, self._batch_shape, self._rand))
                p.start()
                self._sampler_process = p
                self._indices_queue = q
        else:
            print "Sampler process for %s already exists" % self.name
        return self._indices_queue

    def _get_data_queue(self):
        if self._data_queue is None:
            self._data_queue = Queue(self._data_queue_size)
        return self._data_queue

    @classmethod
    def _sampler(cls, indices_queue, dataset, indices_shape, rand):
        handler = Handler()
        try:
            while handler.alive:
                indices = cls._sample_indices(dataset, indices_shape, rand)
                indices_queue.put(indices)
        finally:
            indices_queue.close()
    
    @classmethod
    def _sample_indices(cls, dataset, indices_shape, rand):
        return rand.choice(dataset.size(), indices_shape, replace=False)

    @classmethod
    def _worker(cls, dataset, indices_queue, data_queue, indices_shape, seed):
        handler = Handler()
        # independent random seed
        dataset.rand = np.random.RandomState(seed)
        if indices_queue is None:
            # a placeholder to define the shape of indices
            indices = np.zeros(indices_shape, dtype=np.int32)
        try:
            while handler.alive:
                if indices_queue is not None:
                    indices = indices_queue.get()
                    assert indices.shape == indices_shape
                #
                prefetched_data = dataset.load_data(indices)
                data_queue.put(prefetched_data)
        finally:
            data_queue.close()
            if indices_queue is not None:
                indices_queue.close()

    def get_data(self, indices=None):
        if indices is None:
            if self._prefetch_process_added:
                return self._data_queue.get()
            else:
                indices = self._sample_indices(self.dataset,
                        self._batch_shape, self._rand)
        return self.dataset.load_data(indices)

    def _clean_and_close(self):
        from Queue import Empty
        # first terminate all worker processes
        for wp in self._worker_processes:
            wp.terminate()
        if self._data_queue is not None:
            try:
                while True:
                    self._data_queue.get(timeout=self._timeout)
            except Empty:
                pass
            finally:
                self._data_queue.close()
                self._data_queue.join_thread()
                self._data_queue = None
        for wp in self._worker_processes:
            if wp.is_alive():
                wp.join()
            print "Joined process:", wp.name
        self._worker_processes = list()
        # then terminate the sampler process
        if self._sampler_process is not None:
            self._sampler_process.terminate()
        if self._indices_queue is not None:
            try:
                while True:
                    self._indices_queue.get(timeout=self._timeout)
            except Empty:
                pass
            finally:
                self._indices_queue.close()
                self._indices_queue.join_thread()
                self._indices_queue = None
        if self._sampler_process is not None:
            if self._sampler_process.is_alive():
                self._sampler_process.join()
            print "Joined process:", self._sampler_process.name
            self._sampler_process = None
        self._prefetch_process_added = False
    
    def __del__(self):
        if self._prefetch_process_added:
            self._clean_and_close()
            print "Joined all prefetch processes."

    @staticmethod
    def get_parser(ps=None):
        if ps is None:
            ps = ArgumentParser()
        g = ps.add_argument_group('dataloader')
        #
        g.add_argument('--dl_timeout', type=float, default=1.)
        g.add_argument('--dl_indices_queue_size', type=int, default=10)
        g.add_argument('--dl_data_queue_size', type=int, default=10)
        return ps
