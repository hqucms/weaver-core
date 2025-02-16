from functools import wraps
from itertools import chain
from typing import Any, Callable, List, Literal, Optional, Union

import torch
from torch import Tensor


def minimum_autocast_precision(
    min_dtype: torch.dtype = torch.float32,
    output: Optional[Union[Literal["low", "high"], torch.dtype]] = None,
    which_args: Optional[List[int]] = None,
    which_kwargs: Optional[List[str]] = None,
):
    """Decorator that ensures input tensors are autocast to a minimum precision.
    Only has an effect in autocast-enabled regions. Otherwise, does not change the function.
    Only floating-point inputs are modified. Non-tensors, integer tensors, and boolean tensors are
    untouched.
    Note: AMP is turned on and off separately for CPU and CUDA. This decorator may fail in
    the case where both devices are used, with only one of them on AMP.
    Parameters
    ----------
    min_dtype : dtype
        Minimum dtype. Default: float32.
    output: None or "low" or "high" or dtype
        Specifies which dtypes the outputs should be cast to. Only floating-point Tensor outputs
        are affected. If None, the outputs are not modified. If "low", the lowest-precision input
        dtype is used. If "high", `min_dtype` or the highest-precision input dtype is used
        (whichever is higher).
    which_args : None or list of int
        If not None, specifies which positional arguments are to be modified. If None (the default),
        all positional arguments are modified (if they are Tensors and of a floating-point dtype).
    which_kwargs : bool
        If not None, specifies which keyword arguments are to be modified. If None (the default),
        all keyword arguments are modified (if they are Tensors and of a floating-point dtype).
    Returns
    -------
    decorator : Callable
        Decorator.
    """

    def decorator(func: Callable):
        """Decorator that casts input tensors to minimum precision."""

        def _cast_in(var: Any):
            """Casts a single input to at least 32-bit precision."""
            if not isinstance(var, Tensor):
                # We don't want to modify non-Tensors
                return var
            if not var.dtype.is_floating_point:
                # Integer / boolean tensors are also not touched
                return var
            dtype = max(var.dtype, min_dtype, key=lambda dt: torch.finfo(dt).bits)
            return var.to(dtype)

        def _cast_out(var: Any, dtype: torch.dtype):
            """Casts a single output to desired precision."""
            if not isinstance(var, Tensor):
                # We don't want to modify non-Tensors
                return var
            if not var.dtype.is_floating_point:
                # Integer / boolean tensors are also not touched
                return var
            return var.to(dtype)

        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            """Decorated func."""
            # Only change dtypes in autocast-enabled regions
            if not (torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()):
                # NB: torch.is_autocast_enabled() only checks for GPU autocast
                # See https://github.com/pytorch/pytorch/issues/110966
                return func(*args, **kwargs)
            # Cast inputs to at least 32 bit
            mod_args = [
                _cast_in(arg)
                for i, arg in enumerate(args)
                if which_args is None or i in which_args
            ]
            mod_kwargs = {
                key: _cast_in(val)
                for key, val in kwargs.items()
                if which_kwargs is None or key in which_kwargs
            }
            # Call function w/o autocast enabled
            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(
                device_type="cpu", enabled=False
            ):
                outputs = func(*mod_args, **mod_kwargs)
            # Cast outputs to correct dtype
            if output is None:
                return outputs
            if output in ["low", "high"]:
                in_dtypes = [
                    arg.dtype
                    for arg in chain(args, kwargs.values())
                    if isinstance(arg, Tensor) and arg.dtype.is_floating_point
                ]
                assert len(in_dtypes)
                if output == "low":
                    out_dtype = min(
                        [min_dtype] + in_dtypes, key=lambda dt: torch.finfo(dt).bits
                    )
                else:
                    out_dtype = max(in_dtypes, key=lambda dt: torch.finfo(dt).bits)
            else:
                out_dtype = output
            if isinstance(outputs, tuple):
                return (_cast_out(val, out_dtype) for val in outputs)
            else:
                return _cast_out(outputs, out_dtype)

        return decorated_func

    return decorator
