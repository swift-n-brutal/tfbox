# -*- coding: utf-8 -*-

from argparse import ArgumentParser

class Dataset(object):
    """Dataset
    """
    def size(self):
        raise NotImplementedError

    def load_data(self, indices, blobs=None):
        """
        indices : numpy.ndarray
            Indices of data to load.
        blobs : dict
            Pairs of blob_name and blob_array. Potentially useful for storing
            prefetched data in the specified memory space.
        """
        raise NotImplementedError

    @staticmethod
    def get_parser(ps=None):
        if ps is None:
            ps = ArgumentParser()
        return ps
