import os
import copy
import json
import numpy as np
import awkward as ak
import torch.utils.data

from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor
from .logger import _logger
from .data.tools import _pad, _repeat_pad, _clip, _stack
from .data.fileio import _read_files
from .data.config import DataConfig, _md5
from .data.preprocess import _apply_selection, _build_new_variables, _build_weights, AutoStandardizer, WeightMaker


def _collate_awkward_array_fn(batch, *, collate_fn_map=None):
    return _stack(batch, axis=0)


def _finalize_inputs(table, data_config):
    output = {}
    # copy observer variables before transformation
    for k in data_config.z_variables:
        if k in data_config.observer_names:
            output[k] = table[k]  # ak.Array
    # copy labels
    for k in data_config.label_names:
        output[k] = ak.to_numpy(table[k])
    # transformation
    for k, params in data_config.preprocess_params.items():
        if data_config._auto_standardization and params['center'] == 'auto':
            raise ValueError('No valid standardization params for %s' % k)
        if params['center'] is not None:
            table[k] = _clip((table[k] - params['center']) * params['scale'], params['min'], params['max'])
        if params['length'] is not None:
            pad_fn = _repeat_pad if params['pad_mode'] == 'wrap' else partial(_pad, value=params['pad_value'])
            table[k] = pad_fn(table[k], params['length'])
        # check for NaN
        if np.any(np.isnan(table[k])):
            _logger.warning(
                'Found NaN in %s, silently converting it to 0.', k)
            table[k] = np.nan_to_num(table[k])
    # stack variables for each input group
    for k, names in data_config.input_dicts.items():
        if len(names) == 1 and data_config.preprocess_params[names[0]]['length'] is None:
            output['_' + k] = ak.to_numpy(ak.values_astype(table[names[0]], 'float32'))
        else:
            output['_' + k] = ak.to_numpy(np.stack([ak.to_numpy(table[n]).astype('float32') for n in names], axis=1))
    # copy monitor variables (after transformation)
    for k in data_config.z_variables:
        if k in data_config.monitor_variables:
            output[k] = table[k]  # ak.Array
    return output


def _get_reweight_indices(weights, up_sample=True, max_resample=10, weight_scale=1):
    all_indices = np.arange(len(weights))
    randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights))
    keep_flags = randwgt < weights
    if not up_sample:
        keep_indices = all_indices[keep_flags]
    else:
        n_repeats = len(weights) // max(1, int(keep_flags.sum()))
        if n_repeats > max_resample:
            n_repeats = max_resample
        all_indices = np.repeat(np.arange(len(weights)), n_repeats)
        randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights) * n_repeats)
        keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]
    return copy.deepcopy(keep_indices)


def _check_labels(table):
    if np.all(table['_labelcheck_'] == 1):
        return
    else:
        if np.any(table['_labelcheck_'] == 0):
            raise RuntimeError('Inconsistent label definition: some of the entries are not assigned to any classes!')
        if np.any(table['_labelcheck_'] > 1):
            raise RuntimeError('Inconsistent label definition: some of the entries are assigned to multiple classes!')


def _preprocess(table, data_config, options):
    # apply selection
    table = _apply_selection(
        table, data_config.selection if options['training'] else data_config.test_time_selection,
        funcs=data_config.var_funcs)
    if len(table) == 0:
        return []
    # define new variables
    aux_branches = data_config.train_aux_branches if options['training'] else data_config.test_aux_branches
    table = _build_new_variables(table, {k: v for k, v in data_config.var_funcs.items() if k in aux_branches})
    # check labels
    if data_config.label_type == 'simple' and options['training']:
        _check_labels(table)
    # compute reweight indices
    if options['reweight'] and data_config.weight_name is not None:
        wgts = _build_weights(table, data_config)
        indices = _get_reweight_indices(wgts, up_sample=options['up_sample'],
                                        weight_scale=options['weight_scale'], max_resample=options['max_resample'])
    else:
        indices = np.arange(len(table[data_config.label_names[0]]))
    # shuffle
    if options['shuffle']:
        np.random.shuffle(indices)
    # perform input variable standardization, clipping, padding and stacking
    table = _finalize_inputs(table, data_config)
    return table, indices


def _load_next(data_config, filelist, load_range, options):
    load_branches = data_config.train_load_branches if options['training'] else data_config.test_load_branches
    table = _read_files(filelist, load_branches, load_range, treename=data_config.treename,
                        branch_magic=data_config.branch_magic, file_magic=data_config.file_magic)
    table, indices = _preprocess(table, data_config, options)
    return table, indices


class _SimpleIter(object):
    r"""_SimpleIter

    Iterator object for ``SimpleIterDataset''.
    """

    def __init__(self, **kwargs):
        # inherit all properties from SimpleIterDataset
        self.__dict__.update(**kwargs)

        # executor to read files and run preprocessing asynchronously
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        # init: prefetch holds table and indices for the next fetch
        self.prefetch = None
        self.table = None
        self.indices = []
        self.cursor = 0

        self._seed = None
        worker_info = torch.utils.data.get_worker_info()
        file_dict = copy.deepcopy(self._init_file_dict)
        if worker_info is not None:
            # in a worker process
            self._name += '_worker%d' % worker_info.id
            self._seed = worker_info.seed & 0xFFFFFFFF
            np.random.seed(self._seed)
            # split workload by files
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[worker_info.id::worker_info.num_workers]
                assert (len(new_files) > 0)
                new_file_dict[name] = new_files
            file_dict = new_file_dict
        self.worker_file_dict = file_dict
        self.worker_filelist = sum(file_dict.values(), [])
        self.worker_info = worker_info
        self.restart()

    def restart(self):
        print('=== Restarting DataIter %s, seed=%s ===' % (self._name, self._seed))
        # re-shuffle filelist and load range if for training
        filelist = copy.deepcopy(self.worker_filelist)
        if self._sampler_options['shuffle']:
            np.random.shuffle(filelist)
        if self._file_fraction < 1:
            num_files = int(len(filelist) * self._file_fraction)
            filelist = filelist[:num_files]
        self.filelist = filelist

        if self._init_load_range_and_fraction is None:
            self.load_range = (0, 1)
        else:
            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._sampler_options['shuffle']:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)

        _logger.debug(
            'Init iter [%d], will load %d (out of %d*%s=%d) files with load_range=%s:\n%s', 0
            if self.worker_info is None else self.worker_info.id, len(self.filelist),
            len(sum(self._init_file_dict.values(), [])),
            self._file_fraction, int(len(sum(self._init_file_dict.values(), [])) * self._file_fraction),
            str(self.load_range),
            '\n'.join(self.filelist[: 3]) + '\n ... ' + self.filelist[-1],)

        _logger.info('Restarted DataIter %s, load_range=%s, file_list:\n%s' %
                     (self._name, str(self.load_range), json.dumps(self.worker_file_dict, indent=2)))

        # reset file fetching cursor
        self.ipos = 0 if self._fetch_by_files else self.load_range[0]
        # prefetch the first entry asynchronously
        self._try_get_next(init=True)

    def __next__(self):
        # print(self.ipos, self.cursor)
        if len(self.filelist) == 0:
            raise StopIteration
        try:
            i = self.indices[self.cursor]
        except IndexError:
            # case 1: first entry, `self.indices` is still empty
            # case 2: running out of entries, `self.indices` is not empty
            while True:
                if self._in_memory and len(self.indices) > 0:
                    # only need to re-shuffle the indices, if this is not the first entry
                    if self._sampler_options['shuffle']:
                        np.random.shuffle(self.indices)
                    break
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
                # check if any entries are fetched (i.e., passing selection) -- if not, do another fetch
                if len(self.indices) > 0:
                    break
            # reset cursor
            self.cursor = 0
            i = self.indices[self.cursor]
        self.cursor += 1
        return self.get_data(i)

    def _try_get_next(self, init=False):
        end_of_list = self.ipos >= len(self.filelist) if self._fetch_by_files else self.ipos >= self.load_range[1]
        if end_of_list:
            if init:
                raise RuntimeError('Nothing to load for worker %d' %
                                   0 if self.worker_info is None else self.worker_info.id)
            if self._infinity_mode and not self._in_memory:
                # infinity mode: re-start
                self.restart()
                return
            else:
                # finite mode: set prefetch to None, exit
                self.prefetch = None
                return

        if self._fetch_by_files:
            filelist = self.filelist[int(self.ipos): int(self.ipos + self._fetch_step)]
            load_range = self.load_range
        else:
            filelist = self.filelist
            load_range = (self.ipos, min(self.ipos + self._fetch_step, self.load_range[1]))

        # _logger.info('Start fetching next batch, len(filelist)=%d, load_range=%s'%(len(filelist), load_range))
        if self._async_load:
            self.prefetch = self.executor.submit(_load_next, self._data_config,
                                                 filelist, load_range, self._sampler_options)
        else:
            self.prefetch = _load_next(self._data_config, filelist, load_range, self._sampler_options)
        self.ipos += self._fetch_step

    def get_data(self, i):
        # inputs
        X = {k: copy.deepcopy(self.table['_' + k][i]) for k in self._data_config.input_names}
        # labels
        y = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.label_names}
        # observers / monitor variables
        Z = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.z_variables}
        return X, y, Z


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.

    Handles dataloading.

    Arguments:
        file_dict (dict): dictionary of lists of files to be loaded.
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
        file_fraction (float): fraction of files to load.
    """

    def __init__(self, file_dict, data_config_file,
                 for_training=True, load_range_and_fraction=None, extra_selection=None,
                 fetch_by_files=False, fetch_step=0.01, file_fraction=1, remake_weights=False, up_sample=True,
                 weight_scale=1, max_resample=10, async_load=True, infinity_mode=False, in_memory=False, name=''):
        self._iters = {} if infinity_mode or in_memory else None
        _init_args = set(self.__dict__.keys())
        self._init_file_dict = file_dict
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        self._fetch_step = fetch_step
        self._file_fraction = file_fraction
        self._async_load = async_load
        self._infinity_mode = infinity_mode
        self._in_memory = in_memory
        self._name = name

        # ==== sampling parameters ====
        self._sampler_options = {
            'up_sample': up_sample,
            'weight_scale': weight_scale,
            'max_resample': max_resample,
        }

        # ==== torch collate_fn map ====
        from torch.utils.data._utils.collate import default_collate_fn_map
        default_collate_fn_map.update({ak.Array: _collate_awkward_array_fn})

        if for_training:
            self._sampler_options.update(training=True, shuffle=True, reweight=True)
        else:
            self._sampler_options.update(training=False, shuffle=False, reweight=False)

        # discover auto-generated reweight file
        if '.auto.yaml' in data_config_file:
            data_config_autogen_file = data_config_file
        else:
            data_config_md5 = _md5(data_config_file)
            data_config_autogen_file = data_config_file.replace('.yaml', '.%s.auto.yaml' % data_config_md5)
            if os.path.exists(data_config_autogen_file):
                data_config_file = data_config_autogen_file
                _logger.info('Found file %s w/ auto-generated preprocessing information, will use that instead!' %
                             data_config_file)

        # load data config (w/ observers now -- so they will be included in the auto-generated yaml)
        self._data_config = DataConfig.load(data_config_file)

        if for_training:
            # produce variable standardization info if needed
            if self._data_config._missing_standardization_info:
                s = AutoStandardizer(file_dict, self._data_config)
                self._data_config = s.produce(data_config_autogen_file)

            # produce reweight info if needed
            if self._sampler_options['reweight'] and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
                if remake_weights or self._data_config.reweight_hists is None:
                    w = WeightMaker(file_dict, self._data_config)
                    self._data_config = w.produce(data_config_autogen_file)

            # reload data_config w/o observers for training
            if os.path.exists(data_config_autogen_file) and data_config_file != data_config_autogen_file:
                data_config_file = data_config_autogen_file
                _logger.info(
                    'Found file %s w/ auto-generated preprocessing information, will use that instead!' %
                    data_config_file)
            self._data_config = DataConfig.load(data_config_file, load_observers=False, extra_selection=extra_selection)
        else:
            self._data_config = DataConfig.load(
                data_config_file, load_reweight_info=False, extra_test_selection=extra_selection)

        # derive all variables added to self.__dict__
        self._init_args = set(self.__dict__.keys()) - _init_args

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        if self._iters is None:
            kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
            return _SimpleIter(**kwargs)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            try:
                return self._iters[worker_id]
            except KeyError:
                kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
                self._iters[worker_id] = _SimpleIter(**kwargs)
                return self._iters[worker_id]
