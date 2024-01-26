import numpy as np
import yaml
import copy

from ..logger import _logger
from .tools import _get_variable_names


def _as_list(x):
    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def _md5(fname):
    '''https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file'''
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DataConfig(object):
    r"""Data loading configuration.
    """

    def __init__(self, print_info=True, **kwargs):

        opts = {
            'treename': None,
            'branch_magic': None,
            'file_magic': None,
            'selection': None,
            'test_time_selection': None,
            'preprocess': {'method': 'manual', 'data_fraction': 0.1, 'params': None},
            'new_variables': {},
            'inputs': {},
            'labels': {},
            'observers': [],
            'monitor_variables': [],
            'weights': None,
        }
        for k, v in kwargs.items():
            if v is not None:
                if isinstance(opts[k], dict):
                    opts[k].update(v)
                else:
                    opts[k] = v
        # only information in ``self.options'' will be persisted when exporting to YAML
        self.options = opts
        if print_info:
            _logger.debug(opts)

        self.train_load_branches = set()
        self.train_aux_branches = set()
        self.test_load_branches = set()
        self.test_aux_branches = set()

        self.selection = opts['selection']
        self.test_time_selection = opts['test_time_selection'] if opts['test_time_selection'] else self.selection
        self.var_funcs = copy.deepcopy(opts['new_variables'])
        # preprocessing config
        self.preprocess = opts['preprocess']
        self._auto_standardization = opts['preprocess']['method'].lower().startswith('auto')
        self._missing_standardization_info = False
        self.preprocess_params = opts['preprocess']['params'] if opts['preprocess']['params'] is not None else {}
        # inputs
        self.input_names = tuple(opts['inputs'].keys())
        self.input_dicts = {k: [] for k in self.input_names}
        self.input_shapes = {}
        for k, o in opts['inputs'].items():
            self.input_shapes[k] = (-1, len(o['vars']), o['length'])
            for v in o['vars']:
                v = _as_list(v)
                self.input_dicts[k].append(v[0])

                if opts['preprocess']['params'] is None:

                    def _get(idx, default):
                        try:
                            return v[idx]
                        except IndexError:
                            return default

                    params = {'length': o['length'], 'pad_mode': o.get('pad_mode', 'constant').lower(),
                              'center': _get(1, 'auto' if self._auto_standardization else None),
                              'scale': _get(2, 1), 'min': _get(3, -5), 'max': _get(4, 5), 'pad_value': _get(5, 0)}
                    if v[0] in self.preprocess_params and params != self.preprocess_params[v[0]]:
                        raise RuntimeError(
                            'Incompatible info for variable %s, had: \n  %s\nnow got:\n  %s' %
                            (v[0], str(self.preprocess_params[v[0]]), str(params)))
                    if k.endswith('_mask') and params['pad_mode'] != 'constant':
                        raise RuntimeError('The `pad_mode` must be set to `constant` for the mask input `%s`' % k)
                    if params['center'] == 'auto':
                        self._missing_standardization_info = True
                    self.preprocess_params[v[0]] = params
        # labels
        self.label_type = opts['labels']['type']
        self.label_value = opts['labels']['value']
        if self.label_type == 'simple':
            assert (isinstance(self.label_value, list))
            self.label_names = ('_label_',)
            label_exprs = ['ak.to_numpy(%s)' % k for k in self.label_value]
            self.register('_label_', 'np.argmax(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs)))
            self.register('_labelcheck_', 'np.sum(np.stack([%s], axis=1), axis=1)' % (','.join(label_exprs)), 'train')
        else:
            self.label_names = tuple(self.label_value.keys())
            self.register(self.label_value)
        self.basewgt_name = '_basewgt_'
        self.weight_name = None
        if opts['weights'] is not None:
            self.weight_name = '_weight_'
            self.use_precomputed_weights = opts['weights']['use_precomputed_weights']
            if self.use_precomputed_weights:
                self.register(self.weight_name, '*'.join(opts['weights']['weight_branches']), 'train')
            else:
                self.reweight_method = opts['weights']['reweight_method']
                self.reweight_basewgt = opts['weights'].get('reweight_basewgt', None)
                if self.reweight_basewgt:
                    self.register(self.basewgt_name, self.reweight_basewgt, 'train')
                self.reweight_branches = tuple(opts['weights']['reweight_vars'].keys())
                self.reweight_bins = tuple(opts['weights']['reweight_vars'].values())
                self.reweight_classes = tuple(opts['weights']['reweight_classes'])
                self.register(self.reweight_branches + self.reweight_classes, to='train')
                self.class_weights = opts['weights'].get('class_weights', None)
                if self.class_weights is None:
                    self.class_weights = np.ones(len(self.reweight_classes))
                self.reweight_threshold = opts['weights'].get('reweight_threshold', 10)
                self.reweight_discard_under_overflow = opts['weights'].get('reweight_discard_under_overflow', True)
                self.reweight_hists = opts['weights'].get('reweight_hists', None)
                if self.reweight_hists is not None:
                    for k, v in self.reweight_hists.items():
                        self.reweight_hists[k] = np.array(v, dtype='float32')
        # observers
        self.observer_names = tuple(opts['observers'])
        # monitor variables
        self.monitor_variables = tuple(opts['monitor_variables'])
        if self.observer_names and self.monitor_variables:
            raise RuntimeError('Cannot set `observers` and `monitor_variables` at the same time.')
        # Z variables: returned as `Z` in the dataloader (use monitor_variables for training, observers for eval)
        self.z_variables = self.observer_names if len(self.observer_names) > 0 else self.monitor_variables

        # remove self mapping from var_funcs
        for k, v in self.var_funcs.items():
            if k == v:
                del self.var_funcs[k]

        if print_info:
            def _log(msg, *args, **kwargs):
                _logger.info(msg, *args, color='lightgray', **kwargs)
            _log('preprocess config: %s', str(self.preprocess))
            _log('selection: %s', str(self.selection))
            _log('test_time_selection: %s', str(self.test_time_selection))
            _log('var_funcs:\n - %s', '\n - '.join(str(it) for it in self.var_funcs.items()))
            _log('input_names: %s', str(self.input_names))
            _log('input_dicts:\n - %s', '\n - '.join(str(it) for it in self.input_dicts.items()))
            _log('input_shapes:\n - %s', '\n - '.join(str(it) for it in self.input_shapes.items()))
            _log('preprocess_params:\n - %s', '\n - '.join(str(it) for it in self.preprocess_params.items()))
            _log('label_names: %s', str(self.label_names))
            _log('observer_names: %s', str(self.observer_names))
            _log('monitor_variables: %s', str(self.monitor_variables))
            if opts['weights'] is not None:
                if self.use_precomputed_weights:
                    _log('weight: %s' % self.var_funcs[self.weight_name])
                else:
                    for k in ['reweight_method', 'reweight_basewgt', 'reweight_branches', 'reweight_bins',
                              'reweight_classes', 'class_weights', 'reweight_threshold',
                              'reweight_discard_under_overflow']:
                        _log('%s: %s' % (k, getattr(self, k)))

        # selection
        if self.selection:
            self.register(_get_variable_names(self.selection), to='train')
        # test time selection
        if self.test_time_selection:
            self.register(_get_variable_names(self.test_time_selection), to='test')
        # inputs
        for names in self.input_dicts.values():
            self.register(names)
        # observers
        self.register(self.observer_names, to='test')
        # monitor variables
        self.register(self.monitor_variables)
        # resolve dependencies
        func_vars = set(self.var_funcs.keys())
        for (load_branches, aux_branches) in (self.train_load_branches, self.train_aux_branches), (self.test_load_branches, self.test_aux_branches):
            while (load_branches & func_vars):
                for k in (load_branches & func_vars):
                    aux_branches.add(k)
                    load_branches.remove(k)
                    load_branches.update(_get_variable_names(self.var_funcs[k]))
        if print_info:
            _logger.debug('train_load_branches:\n  %s', ', '.join(sorted(self.train_load_branches)))
            _logger.debug('train_aux_branches:\n  %s', ', '.join(sorted(self.train_aux_branches)))
            _logger.debug('test_load_branches:\n  %s', ', '.join(sorted(self.test_load_branches)))
            _logger.debug('test_aux_branches:\n  %s', ', '.join(sorted(self.test_aux_branches)))

    def __getattr__(self, name):
        return self.options[name]

    def register(self, name, expr=None, to='both'):
        assert to in ('train', 'test', 'both')
        if isinstance(name, dict):
            for k, v in name.items():
                self.register(k, v, to)
        elif isinstance(name, (list, tuple)):
            for k in name:
                self.register(k, None, to)
        else:
            if to in ('train', 'both'):
                self.train_load_branches.add(name)
            if to in ('test', 'both'):
                self.test_load_branches.add(name)
            if expr:
                self.var_funcs[name] = expr
                if to in ('train', 'both'):
                    self.train_aux_branches.add(name)
                if to in ('test', 'both'):
                    self.test_aux_branches.add(name)

    def dump(self, fp):
        with open(fp, 'w') as f:
            yaml.safe_dump(self.options, f, sort_keys=False)

    @classmethod
    def load(cls, fp, load_observers=True, load_reweight_info=True, extra_selection=None, extra_test_selection=None):
        with open(fp) as f:
            _opts = yaml.safe_load(f)
            options = copy.deepcopy(_opts)
        if not load_observers:
            options['observers'] = None
        if not load_reweight_info:
            options['weights'] = None
        if extra_selection:
            options['selection'] = '(%s) & (%s)' % (_opts['selection'], extra_selection)
        if extra_test_selection:
            if 'test_time_selection' not in options or options['test_time_selection'] is None:
                options['test_time_selection'] = '(%s) & (%s)' % (_opts['selection'], extra_test_selection)
            else:
                options['test_time_selection'] = '(%s) & (%s)' % (_opts['test_time_selection'], extra_test_selection)
        return cls(**options)

    def copy(self):
        return self.__class__(print_info=False, **copy.deepcopy(self.options))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def export_json(self, fp):
        import json
        j = {'output_names': self.label_value, 'input_names': self.input_names}
        for k, v in self.input_dicts.items():
            j[k] = {'var_names': v, 'var_infos': {}}
            for var_name in v:
                j[k]['var_length'] = self.preprocess_params[var_name]['length']
                info = self.preprocess_params[var_name]
                j[k]['var_infos'][var_name] = {
                    'median': 0 if info['center'] is None else info['center'],
                    'norm_factor': info['scale'],
                    'replace_inf_value': 0,
                    'lower_bound': -1e32 if info['center'] is None else info['min'],
                    'upper_bound': 1e32 if info['center'] is None else info['max'],
                    'pad': info['pad_value']
                }
        with open(fp, 'w') as f:
            json.dump(j, f, indent=2)
