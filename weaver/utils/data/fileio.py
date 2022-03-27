import math
import awkward as ak
import tqdm
import traceback
from .tools import _concat
from ..logger import _logger


def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop].astype('float32')
    return ak.Array(outputs)


def _read_root(filepath, branches, load_range=None, treename=None):
    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    'Need to specify `treename` as more than one trees are found in file %s: %s' %
                    (filepath, str(branches)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return ak.values_astype(outputs, 'float32')


def _read_awkd(filepath, branches, load_range=None):
    import awkward0
    with awkward0.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = ak.from_awkward0(v[start:stop].astype('float32'))
    return ak.Array(outputs)


def _read_parquet(filepath, branches, load_range=None):
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return ak.values_astype(outputs, 'float32')


def _read_files(filelist, branches, load_range=None, show_progressbar=False, **kwargs):
    import os
    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd', '.parquet'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range, treename=kwargs.get('treename', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
            elif ext == '.parquet':
                a = _read_parquet(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if a is not None:
            table.append(a)
    table = _concat(table)  # ak.Array
    if len(table) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        tree = fout.mktree(treename, {k: v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            tree.extend({k: v[start:start + step] for k, v in table.items()})
            start += step
