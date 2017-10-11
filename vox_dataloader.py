# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 17:15:03 2017

@author: shiwu_001
"""

import os
import os.path as osp
import numpy as np
from scipy import misc, ndimage, io
from config import NP_DTYPE as DTYPE

class VoxDataLoader(object):
    def __init__(self, folder, vox_size, classes, phase,
                 transformer=None, seed=None,
                 full_name='full_vox', part_name='part_vox'):
        self.rand = np.random.RandomState(seed)
        self.folder = folder
        self.vox_size = vox_size
        self.classes, self.class2id = self._init_class_list(classes)
        self.phase = phase
        self.transformer = transformer
        self.full_name = full_name
        self.part_name = part_name
        self._init_vox_list()
        
    def _init_class_list(self, classes):
        fin = file(classes, 'r')
        class_list = list()
        class_dict = dict()
        print 'Load class names'
        for line in fin:
            c = line.strip('\n')
            class_dict[c] = len(class_list)
            class_list.append(c)
            print '|\t', c, class_dict[c]
        return class_list, class_dict
        
    def _init_vox_list(self):
        names = list()
        labels = list()
        print 'Load vox names'
        for c in self.classes:
            root = osp.join(self.folder, str(self.vox_size), c, self.phase)
            flist = os.listdir(root)
            classid = self.class2id[c]
            for fname in flist:
                if fname.endswith('mat'):
                    names.append(fname)
                    labels.append(classid)
        self.n_samples = len(names)
        print self.n_samples, 'samples in total'
        self.names = names
        self.labels = labels
    
    def _load_batch(self, blobs, blob_names, batchids):
        full_name = self.full_name
        part_name = self.part_name
        root = osp.join(self.folder, str(self.vox_size))
        for i, index in enumerate(batchids):
            label = self.labels[index]
            name = self.names[index]
            path = osp.join(root, self.classes[label], self.phase, name)
            mat = io.loadmat(path, variable_names=(full_name, part_name))
            full_vox = mat[full_name].transpose(2,1,0).astype(DTYPE)
            part_vox = mat[part_name].transpose(2,1,0).astype(DTYPE)
            part_vox[part_vox == -1] = DTYPE(0.5)
            blobs[0][i,:,:,:,0] = full_vox
            blobs[1][i,:,:,:,0] = part_vox
            blobs[2][i] = DTYPE(label)
    
    def sample_batchids(self, batchsize):
        #return self.rand.randint(self.n_samples, size=batchsize)
        # without replacement
        return self.rand.choice(self.n_samples, size=batchsize, replace=False)
        
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) == 3, 'VoxDataLoader fills three input blobs (%d given)' % (len(blobs))
        #assert len(blob_names) == 3, 'VoxDataLoader fills three input blobs (%d given)' % (len(blob_names))
        if batchids is None:
            batchids = self.sample_batchids(blobs[0].shape[0])
        self._load_batch(blobs, blob_names, batchids)
        return batchids
        

from multiprocessing import Queue, Process
class VoxDataLoaderPrefetch(VoxDataLoader):
    def __init__(self, queue_size, folder, vox_size, classes, phase,
            transformer=None, seed=None):
        super(VoxDataLoaderPrefetch, self).__init__(
                folder, vox_size, classes, phase,
                transformer=transformer, seed=seed)
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
                        self.folder, self.vox_size, self.classes, self.phase,
                        self.names, self.labels, self.full_name, self.part_name,
                        self.transformer))
            wp.start()
            self.worker_processes.append(wp)

    def _init_batchids_process(self, batchsize):
        if self.batchids_process is not None:
            print 'Batchids process already exists'
            return
        self.batchids_queue = Queue(self.batchids_queue_size)
        self.batchids_process = Process(target=self._batchids_process,
                args=(self.rand, self.n_samples, batchsize, self.batchids_queue))
        self.batchids_process.start()

    @classmethod
    def _batchids_process(cls, rand, n_samples, batchsize, batchids_queue):
        while True:
            #batchids_queue.put(rand.randint(n_samples, size=batchsize))
            # without replacement
            batchids_queue.put(rand.choice(n_samples, size=batchsize, replace=False))

    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
            folder, vox_size, classes, phase, names, labels, full_name, part_name, transformer):
        prefetch_data_full = np.zeros(data_shape, dtype=DTYPE)
        prefetch_data_part = np.zeros(data_shape, dtype=DTYPE)
        prefetch_labels = np.zeros(data_shape[0], dtype=DTYPE)
        root = osp.join(folder, str(vox_size))
        while True:
            batchids = batchids_queue.get()
            for i, index in enumerate(batchids):
                label = labels[index]
                name = names[index]
                path = osp.join(root, classes[label], phase, name)
                mat = io.loadmat(path, variable_names=(full_name, part_name))
                full_vox = mat[full_name].transpose(2,1,0).astype(DTYPE)
                part_vox = mat[part_name].transpose(2,1,0).astype(DTYPE)
                part_vox[part_vox==-1] = DTYPE(0.5)
                prefetch_data_full[i,:,:,:,0] = full_vox
                prefetch_data_part[i,:,:,:,0] = part_vox
                prefetch_labels[i] = DTYPE(label)
            data_queue.put((batchids, (prefetch_data_full.copy(),
                prefetch_data_part.copy(), prefetch_labels.copy())))

    def _get_data(self, blob_name):
        dq = self.data_queue_.get(blob_name)
        assert dq is not None, 'No such blob specified (%s)' % blob_name
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
        assert len(blobs) == 3, 'VoxDataLoader fills three input blobs (%d given)' % (len(blobs))
        #assert len(blob_names) == 3, 'VoxDataLoader fills three input blobs (%d given)' % (len(blob_names))
        if batchids is None:
            batchids, prefetch_set = self._get_data(blob_names[0])
            blobs[0][...] = prefetch_set[0]
            blobs[1][...] = prefetch_set[1]
            blobs[2][...] = prefetch_set[2]
        else:
            self._load_batch(blobs, blob_names, batchids)
        return batchids

def main():
    folder = '/home/sw015/project/gan-recovery/data/ModelNet10'
    vox_size=30
    classes = '/home/sw015/project/gan-recovery/data/ModelNet10/classes.txt'
    phase = 'train'
    vdl = VoxDataLoader(folder, vox_size, classes, phase)

if __name__ == '__main__':
    main()
