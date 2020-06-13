import os
import glob
import copy
import numpy as np
import math
import torch.utils.data

from itertools import chain
from concurrent.futures.thread import ThreadPoolExecutor
from .logger import _logger
from .data.tools import _pad, _clip, _eval_expr
from .data.fileio import _read_files
from .data.config import DataConfig, _md5
from .data.preprocess import _apply_selection, _build_new_variables, _clean_up, AutoStandardizer, WeightMaker

class _SimpleIter(object):
    r"""_SimpleIter

    Iterator object for ``SimpleIterDataset''.
    """

    def __init__(self, **kwargs):
        # inherit all properties from SimpleIterDataset
        self.__dict__.update(**kwargs)

        # executor to read files and run preprocessing asynchronously
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        worker_info = torch.utils.data.get_worker_info()
        filelist = self._init_filelist.copy()
        if worker_info is not None:
            # in a worker process
            np.random.seed(worker_info.seed & 0xFFFFFFFF)
            # split workload by files
            per_worker = int(math.ceil(len(filelist) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            stop = min(start + per_worker, len(filelist))
            filelist = filelist[start:stop]

        if self._shuffle:
            np.random.shuffle(filelist)
        if self._dilation > 1:
            filelist = filelist[::self._dilation]
        self.filelist = filelist

        if self._init_load_range_and_fraction is None:
            self.load_range = (0, 1)
        else:
            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._shuffle:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)

        _logger.debug('Init iter [%d], will load %d (out of %d) files with load_range=%s:\n%s',
                      0 if worker_info is None else worker_info.id,
                      len(self.filelist),
                      len(self._init_filelist) // self._dilation,
                      str(self.load_range),
                      '\n'.join(self.filelist[:3]) + '\n ... ' + self.filelist[-1],
                      )
        # reset iter status
        self.table = None
        self.prefetch = None
        self.ipos = 0 if self._fetch_by_files else self.load_range[0] # fetch cursor
        self.indices = []
        self.cursor = 0
        # prefetch the first entry asynchronously
        self._try_get_next()

    def __next__(self):
        # print(self.ipos, self.cursor)
        if len(self.filelist) == 0:
            raise StopIteration
        try:
            i = self.indices[self.cursor]
        except IndexError:
            if self.prefetch is None:
                # reaching the end as prefetch got nothing
                self.table = None
                if self._async_load:
                    self.executor.shutdown(wait=False)
                raise StopIteration
            # get result from prefetch
            if self._async_load:
                self.table, self.indices = self.prefetch.result()
            else:
                self.table, self.indices = self.prefetch
            # try to load the next ones asynchronously
            self._try_get_next()
            # reset cursor
            self.cursor = 0
            i = self.indices[self.cursor]
        self.cursor += 1
        return self.get_data(i)

    def _load_next(self, filelist, load_range):
        table = _read_files(filelist, self._data_config.load_branches, load_range, treename=self._data_config.treename)
        indices = self.preprocess(table)
        return table, indices

    def _try_get_next(self):
        if self._fetch_by_files:
            if self.ipos >= len(self.filelist):
                self.prefetch = None
                return
            else:
                filelist = self.filelist[int(self.ipos) : int(self.ipos + self._fetch_step)]
                load_range = self.load_range
        else:
            if self.ipos >= self.load_range[1]:
                self.prefetch = None
                return
            else:
                filelist = self.filelist
                load_range = (self.ipos, min(self.ipos + self._fetch_step, self.load_range[1]))
        # _logger.info('Start fetching next batch, len(filelist)=%d, load_range=%s'%(len(filelist), load_range))
        if self._async_load:
            self.prefetch = self.executor.submit(self._load_next, filelist, load_range)
        else:
            self.prefetch = self._load_next(filelist, load_range)
        self.ipos += self._fetch_step

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
                x_indices = np.clip(np.digitize(
                    rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
                y_indices = np.clip(np.digitize(
                    rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
                wgt[pos] = hist[x_indices, y_indices]
            table[self._data_config.weight_name] = wgt

    def finalize_inputs(self, table, preprocess_params):
        for k, params in preprocess_params.items():
            if self._data_config._auto_standardization and params['center'] is None:
                raise ValueError('No valid standardization params for %s' % k)
            if params['center'] is not None:
                table[k] = _clip((table[k] - params['center']) * params['scale'], params['min'], params['max'])
            if params['length'] is not None:
                table[k] = _pad(table[k], params['length'], params['pad_value'])
            # check for NaN
            if np.any(np.isnan(table[k])):
                _logger.warning(
                    'Found NaN in %s, silently converting it to 0.', k)
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
        randwgt = np.random.uniform(
            low=0, high=self._weight_scale, size=len(weights))
        keep_flags = randwgt < weights
        if not self._up_sample:
            keep_indices = all_indices[keep_flags]
        else:
            n_repeats = len(weights) // max(1, int(keep_flags.sum()))
            if n_repeats > self._max_resample:
                n_repeats = self._max_resample
            all_indices = np.repeat(np.arange(len(weights)), n_repeats)
            randwgt = np.random.uniform(
                low=0, high=self._weight_scale, size=len(weights) * n_repeats)
            keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]
        return keep_indices

    def preprocess(self, table):
        # apply selection
        _apply_selection(table, self._data_config.selection)
        # define new variables
        _build_new_variables(table, self._data_config.var_funcs)
        # build weights
        self.build_weights(table)
        # drop unused variables
        _clean_up(table, self._data_config.drop_branches)
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

    def get_data(self, i):
        # inputs
        X = {k: self.table['_' + k][i] for k in self._data_config.input_names}
        # labels
        y = {k: self.table[k][i] for k in self._data_config.label_names}
        # observers
        Z = {k: self.table[k][i] for k in self._data_config.observer_names}
        return X, y, Z


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.

    Handles dataloading.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config_file (str): YAML file containing data format information.
        for_training (bool): flag indicating whether the dataset is used for training or testing.
            When set to ``True``, will enable shuffling and sampling-based reweighting.
            When set to ``False``, will disable shuffling and reweighting, but will load the observer variables.
        load_range_and_fraction (tuple of tuples, ``((start_pos, end_pos), load_frac)``): fractional range of events to load from each file.
            E.g., setting load_range_and_fraction=((0, 0.8), 0.5) will randomly load 50% out of the first 80% events from each file (so load 50%*80% = 40% of the file).
        fetch_by_files (bool): flag to control how events are retrieved each time we fetch data from disk.
            When set to ``True``, will read only a small number (set by ``fetch_step``) of files each time, but load all the events in these files.
            When set to ``False``, will read from all input files, but load only a small fraction (set by ``fetch_step``) of events each time.
            Default is ``False``, which results in a more uniform sample distribution but reduces the data loading speed.
        fetch_step (float or int): fraction of events (when ``fetch_by_files=False``) or number of files (when ``fetch_by_files=True``) to load each time we fetch data from disk.
            Event shuffling and reweighting (sampling) is performed each time after we fetch data.
            So set this to a large enough value to avoid getting an imbalanced minibatch (due to reweighting/sampling), especially when ``fetch_by_files`` set to ``True``.
            Will load all events (files) at once if set to non-positive value.
        dilation (int): file-level reduction factor for file loading.
            Setting dilation=``d`` will load only ``1`` out of every ``d`` files.
    """

    def __init__(self, filelist, data_config_file, for_training=True, load_range_and_fraction=None, fetch_by_files=False, fetch_step=0.01, dilation=1,
                 remake_weights=False, up_sample=True, weight_scale=1, max_resample=10, async_load=True):
        _init_args = set(self.__dict__.keys())
        self._init_filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        if fetch_step <= 0:
            self._fetch_step = len(filelist) if fetch_by_files else 1.
        else:
            self._fetch_step = fetch_step
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
        data_config_autogen_file = data_config_file.replace('.yaml', '.%s.auto.yaml' % data_config_md5)
        if os.path.exists(data_config_autogen_file):
            data_config_file = data_config_autogen_file
            _logger.info('Found file %s w/ auto-generated preprocessing information, will use that instead!' % data_config_file)

        # load data config
        self._data_config = DataConfig.load(data_config_file)

        # produce variable standardization info if needed
        if self._data_config._missing_standardization_info:
            s = AutoStandardizer(filelist, self._data_config)
            self._data_config = s.produce(data_config_autogen_file)

        # produce reweight info if needed
        if self._reweight and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
            if remake_weights or self._data_config.reweight_hists is None:
                w = WeightMaker(filelist, self._data_config)
                self._data_config = w.produce(data_config_autogen_file)
        
        # derive all variables added to self.__dict__
        self._init_args = set(self.__dict__.keys()) - _init_args

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
        return _SimpleIter(**kwargs)
