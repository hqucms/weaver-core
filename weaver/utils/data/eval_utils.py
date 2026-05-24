import math
import functools
import numpy as np
import awkward as ak

from . import tools

_eval_funcs = {"math": math, "np": np, "numpy": np, "ak": ak, "awkward": ak, "len": len}


def _register_funcs(module):
    from inspect import getmembers, isfunction

    _eval_funcs.update(dict(getmembers(module, isfunction)))


@functools.lru_cache(maxsize=None)
def _get_variable_names(expr, exclude=("awkward", "ak", "np", "numpy", "math", "len")):
    import ast

    root = ast.parse(expr)
    return sorted(
        {node.id for node in ast.walk(root) if isinstance(node, ast.Name) and not node.id.startswith("_")}
        - set(exclude)
    )


def _get_data_var_names(expr):
    # Names referenced in `expr` that should be looked up in the data table.
    # Excludes registered functions (np, ak, user helpers from --custom-functions, etc.),
    # which are resolved from `_eval_funcs` at eval time instead.
    return [v for v in _get_variable_names(expr) if v not in _eval_funcs]


@functools.lru_cache(maxsize=None)
def _compile_expr(expr):
    return compile(expr, "<expr>", "eval")


def _eval_expr(expr, table):
    tmp = {**_eval_funcs, **{k: table[k] for k in _get_data_var_names(expr)}}
    return eval(_compile_expr(expr), tmp)


_register_funcs(tools)
