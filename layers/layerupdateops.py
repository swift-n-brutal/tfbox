import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE
from .layer import Layer

class LayerUpdateOps(Layer):
    """Base class of layer with update ops.
    
    Subclass of Layer.

    Attributes:
        update_params (list of tf.tensor): This list contains the update
            parameters. Usually update parameters are not trainable and are
            closely related to data. 
        update_ops_getter (function): The getter returns a list of update ops.
            The absolute name_scope is preserved. Useful to build control
            dependencies and are distinguished from each other in different
            phases.
        update_collection (str): The collection of update ops and params.
    """
    def __init__(self, name='update_ops', update_collection='default'):
        super(LayerUpdateOps, self).__init__(name)
        self.update_params = list()
        self.update_ops_getter = None #lambda *args, **kwargs: list()
        self._update_ops = None
        self.update_collection = update_collection

    def print_info(self, verbose):
        if verbose:
            print self.name
            print '\tIn', self.inputs
            print '\tOut', self.outputs
            print '\tUpdate(%s)' % self.update_collection, self.update_params
