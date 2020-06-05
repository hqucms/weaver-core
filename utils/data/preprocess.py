import os
import time
import glob
import copy
import tqdm
import numpy as np

from collections import defaultdict
from ..logger import _logger
from .tools import _concat, _get_variable_names, _eval_expr
from .fileio import _read_files


def _apply_selection(table, selection):
    if selection is None:
        return
    selected = _eval_expr(selection, table).astype('bool')
    for k in table.keys():
        table[k] = table[k][selected]


def _build_new_variables(table, funcs):
    if funcs is None:
        return
    for k, expr in funcs.items():
        if k in table:
            continue
        table[k] = _eval_expr(expr, table)


def _clean_up(table, drop_branches):
    for k in drop_branches:
        del table[k]


class AutoStandardizer(object):
    r"""AutoStandardizer.

    Class to compute the variable standardization information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        self._filelist = filelist if isinstance(
            filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))

    def read_file(self, filelist):
        self.keep_branches = set()
        self.load_branches = set()
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] in (None, 'auto'):
                self.keep_branches.add(k)
                if k in self._data_config.var_funcs:
                    expr = self._data_config.var_funcs[k]
                    self.load_branches.add(_get_variable_names(expr))
                else:
                    self.load_branches.add(k)
        if self._data_config.selection:
            self.load_branches.update(_get_variable_names(self._data_config.selection))
        _logger.debug('[AutoStandardizer] self.keep_branches:\n  %s', ','.join(self.keep_branches))
        _logger.debug('[AutoStandardizer] self.load_branches:\n  %s', ','.join(self.load_branches))
        table = _read_files(filelist, self.load_branches, self.load_range, show_progressbar=True, treename=self._data_config.treename)
        _apply_selection(table, self._data_config.selection)
        _build_new_variables(table, {k:v for k in self._data_config.var_funcs.items() if k in self.keep_branches})
        _clean_up(table, self.load_branches - self.keep_branches)
        return table

    def make_preprocess_params(self, table):
        _logger.info('Using %d events to calculate standardization info', len(table[list(table.keys())[0]]))
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] in (None, 'auto'):
                a = table[k]
                try:
                    a = a.content
                except AttributeError:
                    pass
                low, center, high = np.percentile(a, [16, 50, 84])
                scale = max(high - center, center - low)
                scale = 1 if scale == 0 else 1. / scale
                params['center'] = float(center)
                params['scale'] = float(scale)
                preprocess_params[k] = params
                _logger.info('[AutoStandardizer] %s low=%s, center=%s, high=%s, scale=%s', k, low, center, high, scale)
        return preprocess_params

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        preprocess_params = self.make_preprocess_params(table)
        self._data_config.preprocess_params = preprocess_params
        # must also propogate the changes to `data_config.options` so it can be persisted
        self._data_config.options['preprocess']['params'] = preprocess_params
        if output:
            _logger.info(
                'Writing YAML file w/ auto-generated preprocessing info to %s' % output)
            self._data_config.dump(output)
        return self._data_config
