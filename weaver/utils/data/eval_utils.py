import math
import numpy as np
import awkward as ak

from . import tools

_eval_funcs = {
    'math': math, 'np': np, 'numpy': np, 'ak': ak, 'awkward': ak, 'len': len
}


def _register_funcs(module):
    from inspect import getmembers, isfunction
    _eval_funcs.update(dict(getmembers(module, isfunction)))


def _get_variable_names(expr, exclude=['awkward', 'ak', 'np', 'numpy', 'math', 'len']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))


def _eval_expr(expr, table):
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update(_eval_funcs)
    return eval(expr, tmp)


_register_funcs(tools)
