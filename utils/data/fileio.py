import numpy as np
from .tools import _concat
from ..logger import _logger


def _read_hdf5(filepath, branches, partial_load=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k:getattr(f.root, k) for k in branches}
    if partial_load is not None and partial_load != (0, 1):
        start, stop = np.trunc(np.asfarray(partial_load) * len(outputs[branches[0]]))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_root(filepath, branches, partial_load=None, treename=None):
    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.decode('utf-8').split(';')[0] for k, v in f.allitems() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError('Need to specify `treename` as more than one trees are found in file %s: %s' % (filepath, str(branches)))
        tree = f[treename]
        if partial_load is not None and partial_load != (0, 1):
            start, stop = np.trunc(np.asfarray(partial_load) * tree.numentries)
        else:
            start, stop = None, None
        outputs = tree.arrays(branches, namedecode='utf-8', entrystart=start, entrystop=stop)
    return outputs


def _read_awkd(filepath, branches, partial_load=None):
    import awkward
    with awkward.load(filepath) as f:
        outputs = {k:f[k] for k in branches}
    if partial_load is not None and partial_load != (0, 1):
        start, stop = np.trunc(np.asfarray(partial_load) * len(outputs[branches[0]]))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_files(filelist, branches, partial_load, **kwargs):
    import os
    from collections import defaultdict
    table = defaultdict(list)
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, partial_load=partial_load)
            elif ext == '.root':
                a = _read_root(filepath, branches, partial_load=partial_load, treename=kwargs.get('treename', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, partial_load=partial_load)
        except Exception as e:
            a = None
            _logger.error(str(e))
        if a is not None:
            for name in branches:
                table[name].append(a[name].astype('float32'))
    table = {name:_concat(arrs) for name, arrs in table.items()}
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.write.compress.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        fout[treename] = uproot.newtree({k:v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            fout[treename].extend({k:v[start:start + step] for k, v in table.items()})
            start += step
