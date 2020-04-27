import os
import time
import glob
import tqdm
import numpy as np

from collections import defaultdict
from ..logger import _logger
from .tools import _concat, _get_variable_names, _eval_expr
from .fileio import _read_awkd, _read_hdf5, _read_root


class WeightMaker(object):
    r"""WeightMaker.

    Class to make reweighting information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config

    def read_file(self, filelist):
        branches = set(self._data_config.reweight_branches) | set(self._data_config.reweight_classes)
        if self._data_config.selection:
            branches.update(_get_variable_names(self._data_config.selection))
        table = defaultdict(list)
        for filepath in tqdm.tqdm(filelist):
            ext = os.path.splitext(filepath)[1]
            if ext == '.h5':
                arrays = _read_hdf5(filepath, branches)
            elif ext == '.root':
                arrays = _read_root(filepath, branches, treename=self._data_config.treename)
            elif ext == '.awkd':
                arrays = _read_awkd(filepath, branches)
            else:
                raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
            if self._data_config.selection:
                selected = _eval_expr(self._data_config.selection, arrays).astype('bool')
            else:
                selected = None
            for name in self._data_config.reweight_branches + self._data_config.reweight_classes:
                table[name].append(arrays[name] if selected is None else arrays[name][selected])
        for name in branches:
            table[name] = _concat(table[name])
        return table

    def make_weights(self, table):
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        # clip variables to be within bin ranges
        table[x_var] = np.clip(table[x_var], min(x_bins), max(x_bins))
        table[y_var] = np.clip(table[y_var], min(y_bins), max(y_bins))

        _logger.info('Using %d events to make weights', len(table[x_var]))

        sum_evts = 0
        max_weight = 0.9
        raw_hists = {}
        class_events = {}
        result = {}
        for label in self._data_config.reweight_classes:
            pos = (table[label] == 1)
            x = table[x_var][pos]
            y = table[y_var][pos]
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            _logger.info('%s:\n %s', label, str(hist.astype('int64')))
            sum_evts += hist.sum()
            raw_hists[label] = hist.astype('float32')
            result[label] = hist.astype('float32')
        if sum_evts != len(table[x_var]):
            _logger.warning('Only %d (out of %d) events actually used to make weights. Check `reweight_classes` definition and `reweight_vars` binnings!', sum_evts, len(table[x_var]))
            time.sleep(5)

        if self._data_config.reweight_method == 'flat':
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                nonzero_vals = hist[hist > 0]
                min_val, med_val = np.min(nonzero_vals), np.median(hist)  # not really used
                ref_val = np.percentile(nonzero_vals, 10)
                _logger.debug('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' % (label, med_val, min_val, ref_val, ref_val / min_val))
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt  # divide by classwgt here will effective increase the weight later
        elif self._data_config.reweight_method == 'ref':
            # use class 0 as the reference
            hist_ref = raw_hists[self._data_config.reweight_classes[0]]
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                upper = np.percentile(ratio[ratio > 0], 90)
                wgt = np.clip(ratio / upper, 0, 1)  # -> [0,1]
                result[label] = wgt
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt  # divide by classwgt here will effective increase the weight later
        # ''equalize'' all classes
        min_nevt = min(class_events.values()) * max_weight  # multiply by max_weight (<1) to add some randomness in the sampling
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / class_events[label]
            result[label] *= class_wgt

        _logger.info('weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str(result[label]))

        _logger.info('Raw hist * weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str((raw_hists[label] * result[label]).astype('int32')))

        return result

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        wgts = self.make_weights(table)
        self._data_config.reweight_hists = wgts
        self._data_config.options['weights']['reweight_hists'] = {k:v.tolist() for k, v in wgts.items()}
        if output:
            _logger.info('Writing YAML file w/ reweighting info to %s' % output)
            self._data_config.dump(output)
        return self._data_config
