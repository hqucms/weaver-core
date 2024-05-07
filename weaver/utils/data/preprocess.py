import time
import glob
import copy
import numpy as np
import awkward as ak

from ..logger import _logger, warn_n_times
from .tools import _get_variable_names, _eval_expr
from .fileio import _read_files


def _apply_selection(table, selection, funcs=None):
    if selection is None:
        return table
    if funcs:
        new_vars = {k: funcs[k] for k in _get_variable_names(selection) if k not in table.fields and k in funcs}
        _build_new_variables(table, new_vars)
    selected = ak.values_astype(_eval_expr(selection, table), 'bool')
    return table[selected]


def _build_new_variables(table, funcs):
    if funcs is None:
        return table
    for k, expr in funcs.items():
        if k in table.fields:
            continue
        table[k] = _eval_expr(expr, table)
    return table


def _build_weights(table, data_config, reweight_hists=None):
    if data_config.weight_name is None:
        raise RuntimeError('Error when building weights: `weight_name` is None!')
    if data_config.use_precomputed_weights:
        return ak.to_numpy(table[data_config.weight_name])
    else:
        x_var, y_var = data_config.reweight_branches
        x_bins, y_bins = data_config.reweight_bins
        rwgt_sel = None
        if data_config.reweight_discard_under_overflow:
            rwgt_sel = (table[x_var] >= min(x_bins)) & (table[x_var] <= max(x_bins)) & \
                (table[y_var] >= min(y_bins)) & (table[y_var] <= max(y_bins))
        # init w/ wgt=0: events not belonging to any class in `reweight_classes` will get a weight of 0 at the end
        wgt = np.zeros(len(table), dtype='float32')
        sum_evts = 0
        if reweight_hists is None:
            reweight_hists = data_config.reweight_hists
        for label, hist in reweight_hists.items():
            pos = table[label] == 1
            if rwgt_sel is not None:
                pos = (pos & rwgt_sel)
            rwgt_x_vals = ak.to_numpy(table[x_var][pos])
            rwgt_y_vals = ak.to_numpy(table[y_var][pos])
            x_indices = np.clip(np.digitize(
                rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
            y_indices = np.clip(np.digitize(
                rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
            wgt[pos] = hist[x_indices, y_indices]
            sum_evts += np.sum(pos)
        if sum_evts != len(table):
            warn_n_times(
                'Not all selected events used in the reweighting. '
                'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
                '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
            )
        if data_config.reweight_basewgt:
            wgt *= ak.to_numpy(table[data_config.basewgt_name])
        return wgt


class AutoStandardizer(object):
    r"""AutoStandardizer.

    Class to compute the variable standardization information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(
            filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))

    def read_file(self, filelist):
        keep_branches = set()
        aux_branches = set()
        load_branches = set()
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                keep_branches.add(k)
                load_branches.add(k)
        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))

        func_vars = set(self._data_config.var_funcs.keys())
        while (load_branches & func_vars):
            for k in (load_branches & func_vars):
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))

        _logger.debug('[AutoStandardizer] keep_branches:\n  %s', ','.join(keep_branches))
        _logger.debug('[AutoStandardizer] aux_branches:\n  %s', ','.join(aux_branches))
        _logger.debug('[AutoStandardizer] load_branches:\n  %s', ','.join(load_branches))

        table = _read_files(filelist, load_branches, self.load_range, show_progressbar=True,
                            treename=self._data_config.treename,
                            branch_magic=self._data_config.branch_magic, file_magic=self._data_config.file_magic)
        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})
        table = table[keep_branches]
        return table

    def make_preprocess_params(self, table):
        _logger.info('Using %d events to calculate standardization info', len(table))
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                if k.endswith('_mask'):
                    params['center'] = None
                else:
                    a = ak.to_numpy(ak.flatten(table[k], axis=None))
                    # check for NaN
                    if np.any(np.isnan(a)):
                        _logger.warning('[AutoStandardizer] Found NaN in `%s`, will convert it to 0.', k)
                        time.sleep(10)
                        a = np.nan_to_num(a)
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
        # must also propagate the changes to `data_config.options` so it can be persisted
        self._data_config.options['preprocess']['params'] = preprocess_params
        if output:
            _logger.info(
                'Writing YAML file w/ auto-generated preprocessing info to %s' % output)
            self._data_config.dump(output)
        return self._data_config


class WeightMaker(object):
    r"""WeightMaker.

    Class to make reweighting information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()

    def read_file(self, filelist):
        keep_branches = set(self._data_config.reweight_branches + self._data_config.reweight_classes)
        if self._data_config.reweight_basewgt:
            keep_branches.add(self._data_config.basewgt_name)
        aux_branches = set()
        load_branches = keep_branches.copy()
        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))

        func_vars = set(self._data_config.var_funcs.keys())
        while (load_branches & func_vars):
            for k in (load_branches & func_vars):
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))

        _logger.debug('[WeightMaker] keep_branches:\n  %s', ','.join(keep_branches))
        _logger.debug('[WeightMaker] aux_branches:\n  %s', ','.join(aux_branches))
        _logger.debug('[WeightMaker] load_branches:\n  %s', ','.join(load_branches))

        table = _read_files(filelist, load_branches, show_progressbar=True,
                            treename=self._data_config.treename,
                            branch_magic=self._data_config.branch_magic, file_magic=self._data_config.file_magic)
        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})
        table = table[keep_branches]
        return table

    def make_weights(self, table):
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        if not self._data_config.reweight_discard_under_overflow:
            # clip variables to be within bin ranges
            x_min, x_max = min(x_bins), max(x_bins)
            y_min, y_max = min(y_bins), max(y_bins)
            _logger.info(f'Clipping `{x_var}` to [{x_min}, {x_max}] to compute the shapes for reweighting.')
            _logger.info(f'Clipping `{y_var}` to [{y_min}, {y_max}] to compute the shapes for reweighting.')
            table[x_var] = np.clip(table[x_var], min(x_bins), max(x_bins))
            table[y_var] = np.clip(table[y_var], min(y_bins), max(y_bins))

        _logger.info('Using %d events to make weights', len(table))

        sum_evts = 0
        max_weight = 0.9
        raw_hists = {}
        class_events = {}
        result = {}
        for label in self._data_config.reweight_classes:
            pos = (table[label] == 1)
            x = ak.to_numpy(table[x_var][pos])
            y = ak.to_numpy(table[y_var][pos])
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            _logger.info('%s (unweighted):\n %s', label, str(hist.astype('int64')))
            sum_evts += hist.sum()
            if self._data_config.reweight_basewgt:
                w = ak.to_numpy(table[self._data_config.basewgt_name][pos])
                hist, _, _ = np.histogram2d(x, y, weights=w, bins=self._data_config.reweight_bins)
                _logger.info('%s (weighted):\n %s', label, str(hist.astype('float32')))
            raw_hists[label] = hist.astype('float32')
            result[label] = hist.astype('float32')
        if sum_evts != len(table):
            _logger.warning(
                'Only %d (out of %d) events actually used in the reweighting. '
                'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
                '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
                sum_evts, len(table))
            time.sleep(10)

        if self._data_config.reweight_method == 'flat':
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                min_val, med_val = np.min(nonzero_vals), np.median(hist)  # not really used
                ref_val = np.percentile(nonzero_vals, self._data_config.reweight_threshold)
                _logger.debug('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' %
                              (label, med_val, min_val, ref_val, ref_val / min_val))
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        elif self._data_config.reweight_method == 'ref':
            # use class 0 as the reference
            hist_ref = raw_hists[self._data_config.reweight_classes[0]]
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold)
                wgt = np.clip(ratio / upper, 0, 1)  # -> [0,1]
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        # ''equalize'' all classes
        # multiply by max_weight (<1) to add some randomness in the sampling
        min_nevt = min(class_events.values()) * max_weight
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / class_events[label]
            result[label] *= class_wgt

        if self._data_config.reweight_basewgt:
            wgts = _build_weights(table, self._data_config, reweight_hists=result)
            _logger.info('Sample weight percentiles: %s', str(np.percentile(wgts, np.arange(101))))
            wgt_ref = np.percentile(wgts, 100 - self._data_config.reweight_threshold)
            _logger.info('Set overall reweighting scale factor (%d threshold) to %s (max %s)' %
                         (100 - self._data_config.reweight_threshold, wgt_ref, np.max(wgts)))
            for label in self._data_config.reweight_classes:
                result[label] /= wgt_ref

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
        # must also propagate the changes to `data_config.options` so it can be persisted
        self._data_config.options['weights']['reweight_hists'] = {k: v.tolist() for k, v in wgts.items()}
        if output:
            _logger.info('Writing YAML file w/ reweighting info to %s' % output)
            self._data_config.dump(output)
        return self._data_config
