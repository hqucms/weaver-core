import os
import glob
import numpy as np
import math
import torch.utils.data

from itertools import chain
from concurrent.futures.thread import ThreadPoolExecutor
from .logger import _logger
from .data.tools import _pad, _clip, _eval_expr
from .data.fileio import _read_files
from .data.config import DataConfig, _md5
from .data.reweight import WeightMaker


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.

    Handles dataloading.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config_file (str): YAML file containing data format information.
        for_training (bool): flag indicating whether the dataset is used for training or testing.
            When set to ``True``, will enable shuffling and sampling-based reweighting.
            When set to ``False``, will disable shuffling and reweighting, but will load the observer variables.
        partial_load (tuple of tuples, ``((start_pos, end_pos), load_frac)``): fractional range of events to load from each file.
            E.g., setting partial_load=((0, 0.8), 0.5) will randomly load 50% out of the first 80% events from each file (so load 50%*80% = 40% of the file).
        files_per_fetch (int): number of files to load and process each time we fetch data from disk.
            Event shuffling and reweighting (sampling) is performed each time after we fetch data.
            Default is 1, set it to larger values if number of events in each input file is small (e.g., due to reweighting/sampling).
            Will load all files at once if set to non-positive value.
        dilation (int): file-level reduction factor for file loading.
            Setting dilation=``d`` will load only ``1`` out of every ``d`` files.
    """

    def __init__(self, filelist, data_config_file, for_training=True, partial_load=None, files_per_fetch=20, dilation=1,
                 remake_weights=False, up_sample=True, weight_scale=1, max_resample=10, async_load=True):
        self._init_filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._init_partial_load = partial_load
        self._files_per_fetch = files_per_fetch if files_per_fetch > 0 else len(filelist)
        self._dilation = dilation
        self._async_load = async_load

        # ==== sampling parameters ====
        self._up_sample = up_sample
        self._weight_scale = weight_scale
        self._max_resample = max_resample

        if for_training:
            self._shuffle = True
            self._reweight = True
            self._load_observers = False
        else:
            self._shuffle = False
            self._reweight = False
            self._load_observers = True

        # discover auto-generated reweight file
        data_config_md5 = _md5(data_config_file)
        data_config_reweight_file = data_config_file.replace('.yaml', '.%s.reweight.yaml' % data_config_md5)
        if self._reweight and os.path.exists(data_config_reweight_file):
            data_config_file = data_config_reweight_file
            _logger.info('Found file %s w/ reweighting information, will use that instead!' % data_config_file)

        # load data config
        self._data_config = DataConfig.load(data_config_file)

        # produce reweight info if needed
        if self._reweight and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
            if remake_weights or self._data_config.reweight_hists is None:
                w = WeightMaker(filelist, self._data_config)
                w.produce(data_config_reweight_file)

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        # executor to read files and run preprocessing asynchronously
        self._executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        worker_info = torch.utils.data.get_worker_info()
        filelist = self._init_filelist.copy()
        if worker_info is None:
            # single-process data loading, return the full iterator
            pass
        else:
            # in a worker process, split workload
            np.random.seed(worker_info.seed & 0xFFFFFFFF)
            per_worker = int(math.ceil(len(filelist) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            stop = min(start + per_worker, len(filelist))
            filelist = filelist[start:stop]

        if self._shuffle:
            np.random.shuffle(filelist)
        if self._dilation > 1:
            filelist = filelist[::self._dilation]
        self.filelist = filelist
        
        if self._init_partial_load is None:
            self.partial_load = None
        else:
            (start_pos, end_pos), load_frac = self._init_partial_load
            interval = (end_pos - start_pos) * load_frac
            if self._shuffle:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.partial_load = (offset, offset + interval)
            else:
                self.partial_load = (start_pos, start_pos + interval)

        _logger.debug('Init iter [%d], will load %d (out of %d) files with partial_load=%s:\n%s',
                      0 if worker_info is None else worker_info.id, len(self.filelist), len(self._init_filelist) // self._dilation,
                      str(self.partial_load),
                      '\n'.join(self.filelist[:3]) + '\n...\n' + '\n'.join(self.filelist[-3:])
                      )
        # reset iter status
        self.table = None
        self._prefetch = None
        self.ifile = 0
        self.indices = []
        self.cursor = 0
        # prefetch the first entry asynchronously
        self._try_get_next()
        return self

    def __next__(self):
#         print(self.ifile, self.cursor)
        try:
            i = self.indices[self.cursor]
        except IndexError:
            if self._prefetch is None:
                # reaching the end as prefetch got nothing
                self.table = None
                if self._async_load:
                    self._executor.shutdown()
                raise StopIteration
            # get result from prefetch
            if self._async_load:
                self.table, self.indices = self._prefetch.result()
            else:
                self.table, self.indices = self._prefetch
            # try to load the next ones asynchronously
            self._try_get_next()
            # reset cursor
            self.cursor = 0
            i = self.indices[self.cursor]
        self.cursor += 1
        return self.get_data(self.table, i)

    def _load_next(self, filelist):
        table = _read_files(filelist, self._data_config.load_branches, self.partial_load, treename=self._data_config.treename)
        indices = self.preprocess(table)
        return table, indices

    def _try_get_next(self):
        if self.ifile >= len(self.filelist):
            self._prefetch = None
        else:
            filelist = self.filelist[self.ifile:self.ifile + self._files_per_fetch]
            if self._async_load:
                self._prefetch = self._executor.submit(self._load_next, filelist)
            else:
                self._prefetch = self._load_next(filelist)
            self.ifile += self._files_per_fetch

    def apply_selection(self, table, selection):
        if selection is None:
            return
        selected = _eval_expr(selection, table).astype('bool')
        for k in table.keys():
            table[k] = table[k][selected]

    def build_new_variables(self, table, funcs):
        if funcs is None:
            return
        for k, expr in funcs.items():
            if k in table:
                continue
            table[k] = _eval_expr(expr, table)

    def build_weights(self, table):
        if self._reweight and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
            x_var, y_var = self._data_config.reweight_branches
            x_bins, y_bins = self._data_config.reweight_bins
            # init w/ wgt=0: events not belonging to any class in `reweight_classes` will get a weight of 0 at the end
            wgt = np.zeros(len(table[x_var]), dtype='float32')
            for label, hist in self._data_config.reweight_hists.items():
                pos = table[label] == 1
                rwgt_x_vals = table[x_var][pos]
                rwgt_y_vals = table[y_var][pos]
                x_indices = np.clip(np.digitize(rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
                y_indices = np.clip(np.digitize(rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
                wgt[pos] = hist[x_indices, y_indices]
            table[self._data_config.weight_name] = wgt

    def clean_up(self, table, drop_branches):
        for k in drop_branches:
            del table[k]

    def finalize_inputs(self, table, preprocess_params):
        for k, params in preprocess_params.items():
            if params['center'] is not None:
                table[k] = _clip((table[k] - params['center']) * params['scale'], params['min'], params['max'])
            if params['length'] is not None:
                table[k] = _pad(table[k], params['length'], params['pad_value'])
            # check for NaN
            if np.any(np.isnan(table[k])):
                _logger.warning('Found NaN in %s, silently converting it to 0.', k)
                table[k] = np.nan_to_num(table[k])
        # stack variables for each input group
        for k, names in self._data_config.input_dicts.items():
            table['_' + k] = np.stack([table[n] for n in names], axis=1)
        # reduce memory usage
        for n in set(chain(*self._data_config.input_dicts.values())):
            if n not in self._data_config.label_names and n not in self._data_config.observer_names:
                del table[n]

    def get_reweight_indices(self, weights):
        all_indices = np.arange(len(weights))
        randwgt = np.random.uniform(low=0, high=self._weight_scale, size=len(weights))
        keep_flags = randwgt < weights
        if not self._up_sample:
            keep_indices = all_indices[keep_flags]
        else:
            n_repeats = len(weights) // max(1, int(keep_flags.sum()))
            if n_repeats > self._max_resample:
                n_repeats = self._max_resample
            all_indices = np.repeat(np.arange(len(weights)), n_repeats)
            randwgt = np.random.uniform(low=0, high=self._weight_scale, size=len(weights) * n_repeats)
            keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]
        return keep_indices

    def preprocess(self, table):
        # apply selection
        self.apply_selection(table, self._data_config.selection)
        # define new variables
        self.build_new_variables(table, self._data_config.var_funcs)
        # build weights
        self.build_weights(table)
        # drop unused variables
        self.clean_up(table, self._data_config.drop_branches)
        # perform input variable standardization, clipping, padding and stacking
        self.finalize_inputs(table, self._data_config.preprocess_params)
        # compute reweight indices
        if self._reweight and self._data_config.weight_name is not None:
            indices = self.get_reweight_indices(table[self._data_config.weight_name])
        else:
            indices = np.arange(len(table[self._data_config.label_names[0]]))
        # shuffle
        if self._shuffle:
            np.random.shuffle(indices)
        return indices

    def get_data(self, table, i):
        # inputs
        X = {k:table['_' + k][i] for k in self._data_config.input_names}
        # labels
        y = {k:table[k][i] for k in self._data_config.label_names}
        # observers
        Z = {k:table[k][i] for k in self._data_config.observer_names}
        return X, y, Z
