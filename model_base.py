class ModelBase(object):
    def __init__(self):
        self.inputs = list()
        self.outputs = list()
        self.params = list()
        self.param_lrs = list()
        self.param_by_lr_ = dict()
        self.update_ops_ = dict()
        self.update_params_ = dict()

    def _add_params(self, params, lr):
        if len(params) > 0:
            self.params.extend(params)
            if type(lr) is float:
                if self.param_by_lr_.get(lr) is None:
                    self.param_by_lr_[lr] = list()
                self.param_lrs.extend([lr]*len(params))
                self.param_by_lr_[lr].extend(params)
            elif type(lr) is list:
                assert len(lr) == len(params), 'Number of lr and params must be equal (%d != %d)' % (len(lr), len(params))
                self.param_lrs.extend(lr)
                for (l, p) in zip(lr, params):
                    if self.param_by_lr_.get(l) is None:
                        self.param_by_lr_[l] = list()
                    self.param_by_lr_[l].append(p)
            else:
                raise ValueError('The type of lr must be float or list (%s given)' % type(lr))

    def _append(self, layer, lr=1., init=False, update=None):
        if update is not None:
            #assert len(ops) == 4, 'Update module should have 4 components: ops, params, update_ops, update_params (%d given)' % (len(ops))
            if self.update_ops_.get(update) is None:
                self.update_ops_[update] = list()
                self.update_params_[update] = list()
            self.update_ops_[update].extend(layer.update_ops)
            if init:
                self.update_params_[update].extend(layer.update_params)
        if init:
            self._add_params(layer.params, lr)
        return layer.outputs

    def _extend(self, model, init=False, update=None):
        if init:
            self.params.extend(model.params)
            self.param_lrs.extend(model.param_lrs)
            for k in model.param_by_lr_.keys():
                if self.param_by_lr_.get(k) is None:
                    self.param_by_lr_[k] = list()
                self.param_by_lr_[k].extend(model.param_by_lr_[k])
        if update is not None:
            if self.update_ops_.get(update) is None:
                self.update_ops_[update] = list()
                self.update_params_[update] = list()
            self.update_ops_[update].extend(model.update_ops_[update])
            if init:
                self.update_params_[update].extend(model.update_params_[update])

    def get_update_ops(self, name):
        if self.update_ops_.get(name) is None:
            return list()
        return self.update_ops_[name]

    def get_update_params(self, name):
        if self.update_params_.get(name) is None:
            return list()
        return self.update_params_[name]
