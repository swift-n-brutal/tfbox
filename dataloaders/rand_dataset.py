# -*- coding: utf-8 -*-

import numpy as np

from ..config import NP_DTYPE as DTYPE
from .dataset import Dataset

class RandDataset(Dataset):
    """Random number dataset
    """
    def __init__(self, args, dim, seed=None, name='rand_dataloader'):
        self.seed = seed
        self.rand = None
        self.dim = dim
        #
        self._type = args['d_rand_type']
        self._mean = args['d_rand_mean']
        self._std = args['d_rand_std']

    def size(self):
        return 0

    def load_data(self, indices, blobs=None):
        """load a batch of data indexed by indices

        Parameters
        ----------
        indices : np.ndarray
            This should be a placeholder to define the shape of indices.
        blobs : OrderedDict
            The ordered pairs of blob_name and blob_array. Here is only one
            blob for Gram matrices.
        """
        rand = self.rand
        dim = self.dim
        rand_type = self._type
        std = self._std
        mean = self._mean
        shape = indices.shape + (dim,)
        ret = dict()
        if rand_type == 'gaussian':
            ret['data'] = mean + std * rand.standard_normal(shape)
        elif rand_type == 'uniform':
            a = np.sqrt(3) * std
            ret['data'] = mean + rand.uniform(-a, a, size=shape)
        return ret

    @staticmethod
    def get_parser(ps=None):
        ps = super(RandDataset, RandDataset).get_parser(ps)
        #
        g = ps.add_argument_group('rand_dataloader')
        g.add_argument('--d_rand_type', type=str, default='gaussian', choices=['gaussian', 'uniform'])
        g.add_argument('--d_rand_std', type=float, default=1.)
        g.add_argument('--d_rand_mean', type=float, default=0.)
        return ps
