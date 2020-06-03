import numpy as np
import awkward
import math


def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)


def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, awkward.JaggedArray):
        return a.pad(maxlen, clip=True).fillna(value).regular().astype(dtype)
    else:
        x = (np.ones(len(a), maxlen) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    if isinstance(a, np.ndarray):
        return np.clip(a, a_min, a_max)
    else:
        return awkward.JaggedArray.fromcounts(a.counts, np.clip(a.content, a_min, a_max))


def _get_variable_names(expr, exclude=['awkward', 'np', 'numpy', 'math']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, table):
    tmp = {k:table[k] for k in _get_variable_names(expr)}
    tmp.update({'math': math, 'np': np, 'awkward': awkward, '_concat': _concat, '_pad': _pad, '_clip':_clip})
    return eval(expr, tmp)

