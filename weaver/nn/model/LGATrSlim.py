"""L-GATr-slim: a slim Lorentz-equivariant transformer for jet tagging.

A slimmer variant of the Lorentz-Equivariant Geometric Algebra Transformer (L-GATr)
that operates directly on Lorentz vectors and scalars instead of full multivectors.

Papers:
 - "Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics"
   https://arxiv.org/abs/2405.14806
 - "A Guide to Optimal Transformer Architectures for Jet Tagging"
   (tagging-guide, L-GATr-slim reference implementation)

Ported from the `lgatr` package v2.0.0 (https://github.com/heidelberg-hepml/lgatr,
``lgatr/nets/slim.py`` and ``lgatr/layers/slim_layers.py``) and the tagging wrapper of
https://github.com/heidelberg-hepml/tagging-guide (``experiments/tagging/wrappers.py``
and ``experiments/tagging/embedding.py``). The tagging setup uses identity frames, so
none of the LLoCa frames machinery is required. The default attention backend,
``"auto"``, picks per call from the device and the attention precision: on CUDA
``"efficient"`` (exact fp32 packed attention) when attention runs in fp32 and ``"varlen"``
under fp16/bf16 AMP; on CPU (and for ONNX export) ``"native"``, i.e.
``torch.nn.functional.scaled_dot_product_attention`` on the dense zero-padded layout.
``attention_backend="varlen"`` (torch >= 2.10 native flash-attention varlen
kernel), ``"flash"`` (the flash-attn package), ``"xformers"``
(``xformers.ops.memory_efficient_attention`` with a block-diagonal mask), or ``"flex"``
(``torch.nn.attention.flex_attention`` with a block-diagonal ``BlockMask``) instead drop
the padding and run block-diagonal attention over the packed tokens, as in the
tagging-guide sparse path.

The three flash-derived packed backends have no fp32 kernel and therefore run attention
in half precision. In Cartesian coordinates (``vector_coord="cartesian"``) that costs ~1 pp of
tagging accuracy against ``"native"``, while ``"flex"`` has an fp32 path and keeps
``"native"``'s precision; see :func:`_half_cast_dtype` and :func:`_flex_attention`. The
default per-jet light-cone frame (:func:`get_lightcone_frame`) makes half-precision attention
as accurate as fp32, and under AMP every backend then runs attention in the AMP dtype.

That frame is built on the summed constituent momentum, which is a jet assumption. For
**full events** the axis is fitted to the particles instead (``lightcone_frame=
"principal"`` or ``"collinear"``, see :func:`get_principal_axis` and
:func:`get_collinear_axis`); it is still one frame per event, so attention is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial, wraps
from itertools import chain
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from weaver.nn.model.kinematics import (
    get_auxiliary_scalars,
    get_num_auxiliary_scalars,
)
from weaver.nn.model.ParticleTransformer import SequenceTrimmer
from weaver.utils.logger import _logger


# ------------------------------------------------------------------------------------
# Autocast helpers (ported from lgatr/utils/autocast.py)
# ------------------------------------------------------------------------------------

# Toggled by the naive_amp context manager; read at call time so torch.compile constant-folds it.
_NAIVE_AMP = False


try:
    torch.is_autocast_enabled("cpu")

    def _autocast_active(device_type: str | None = None) -> bool:
        """Whether autocast is enabled, for ``device_type`` or for CPU/CUDA at all."""
        if device_type is not None:
            return torch.is_autocast_enabled(device_type)
        return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")

except TypeError:  # pragma: no cover - torch<2.4 has no device_type argument

    def _autocast_active(device_type: str | None = None) -> bool:
        """Whether autocast is enabled, for ``device_type`` or for CPU/CUDA at all."""
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        if device_type == "cuda":
            return torch.is_autocast_enabled()
        return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()


class naive_amp:
    """Disable all :class:`minimum_autocast_precision` pinning inside the block.

    While active, the fp32 precision islands created by the :class:`minimum_autocast_precision`
    decorator are bypassed and the wrapped ops run in the surrounding autocast dtype (e.g. bf16).
    Restores the previous state on exit; safe to nest.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._prev: list[bool] = []

    def __enter__(self) -> "naive_amp":
        global _NAIVE_AMP
        if self.enabled:
            self._prev.append(_NAIVE_AMP)
            _NAIVE_AMP = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _NAIVE_AMP
        if self.enabled:
            _NAIVE_AMP = self._prev.pop()
        return False


class minimum_autocast_precision:
    """Pin tensors to a minimum precision inside autocast regions.

    Used as a decorator: ``@minimum_autocast_precision(torch.float32)``. Inside
    autocast-enabled regions, floating-point inputs below ``min_dtype`` are cast up to
    ``min_dtype``, autocast is disabled for the call, and outputs are optionally cast per the
    ``output`` argument (``"low"``: lowest of ``min_dtype`` and input dtypes; ``"high"``:
    highest input dtype; ``None``: unmodified; or an explicit dtype). Outside autocast
    regions the decorator is a no-op, as it is under :class:`naive_amp`.
    """

    def __init__(
        self,
        min_dtype: torch.dtype = torch.float32,
        output: str | torch.dtype | None = "low",
    ) -> None:
        self.min_dtype = min_dtype
        self.output = output

    def cast(self, var: Any) -> Any:
        """Upcast a floating-point tensor to at least ``min_dtype``."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        if torch.finfo(var.dtype).bits >= torch.finfo(self.min_dtype).bits:
            return var
        return var.to(self.min_dtype)

    def _cast_out(self, var: Any, dtype: torch.dtype) -> Any:
        """Cast a single output to the requested dtype."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        return var.to(dtype)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            # Skip in naive-AMP mode (run in the autocast dtype), or outside autocast regions.
            if _NAIVE_AMP or not _autocast_active():
                return func(*args, **kwargs)
            # Cast inputs to at least min_dtype
            mod_args = [self.cast(arg) for arg in args]
            mod_kwargs = {key: self.cast(val) for key, val in kwargs.items()}
            with (
                torch.autocast(device_type="cuda", enabled=False),
                torch.autocast(device_type="cpu", enabled=False),
            ):
                outputs = func(*mod_args, **mod_kwargs)
            return self._apply_output_dtype(outputs, args, kwargs)

        return decorated_func

    def _apply_output_dtype(self, outputs: Any, args: tuple, kwargs: dict) -> Any:
        """Cast outputs per the ``output`` mode; see class docstring."""
        if self.output is None:
            return outputs
        if self.output in ["low", "high"]:
            in_dtypes = [
                arg.dtype
                for arg in chain(args, kwargs.values())
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point
            ]
            if not in_dtypes:
                return outputs
            # Plain loop instead of min/max(..., key=lambda) to avoid graph breaks in torch.compile
            if self.output == "low":
                candidates = [self.min_dtype] + in_dtypes
                out_dtype = candidates[0]
                for dt in candidates[1:]:
                    if torch.finfo(dt).bits < torch.finfo(out_dtype).bits:
                        out_dtype = dt
            else:
                out_dtype = in_dtypes[0]
                for dt in in_dtypes[1:]:
                    if torch.finfo(dt).bits > torch.finfo(out_dtype).bits:
                        out_dtype = dt
        else:
            out_dtype = self.output
        if isinstance(outputs, tuple):
            return tuple(self._cast_out(val, out_dtype) for val in outputs)
        return self._cast_out(outputs, out_dtype)


def get_nonlinearity(label: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the ``torch.nn.functional`` activation for the given label.

    Accepts ``"relu"``, ``"sigmoid"``, ``"tanh"``, ``"gelu"``, or ``"silu"``.
    ``"gelu"`` uses ``approximate="tanh"``.
    """
    if label == "relu":
        return F.relu
    elif label == "sigmoid":
        return F.sigmoid
    elif label == "tanh":
        return F.tanh
    elif label == "gelu":
        return partial(F.gelu, approximate="tanh")
    elif label == "silu":
        return F.silu
    else:
        raise ValueError(f"Unsupported nonlinearity type: {label}")


# ------------------------------------------------------------------------------------
# Slim layers (ported from lgatr/layers/slim_layers.py)
# ------------------------------------------------------------------------------------


def _require_scalars(**named: torch.Tensor | None) -> None:
    """Raise if any named scalar tensor is None or has zero channels (slim nets require scalars)."""
    for name, tensor in named.items():
        if tensor is None or tensor.shape[-1] == 0:
            raise ValueError(
                f"{name} must be a non-empty scalar tensor; slim networks require scalars."
            )


def _movedim(t: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    # equivalent to torch.movedim, but via permute with normalized (non-negative) dims:
    # negative dims can produce invalid Transpose nodes in the TorchScript ONNX exporter
    n = t.dim()
    src = source % n
    dst = destination % n
    perm = [d for d in range(n) if d != src]
    perm.insert(dst, src)
    return t.permute(perm)


def _apply_metric(w: torch.Tensor, metric: torch.Tensor, lightcone: bool) -> torch.Tensor:
    """``eta @ w`` over the Lorentz-component dim (-2) of ``w`` (..., 4, channels).

    In Cartesian ``(E, px, py, pz)`` coordinates ``eta = diag(1, -1, -1, -1)``. In the
    light-cone coordinates ``(x+, x-, x1, x2)`` of :func:`get_lightcone_frame`, the metric
    pairs ``x+`` with ``x-`` and negates the two transverse components, so ``eta @ w``
    swaps the first two components.
    """
    if lightcone:
        return torch.cat([w[..., 1:2, :], w[..., 0:1, :], -w[..., 2:, :]], dim=-2)
    return w * metric.to(w.dtype)[..., None]


def _lightcone_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Light-cone Minkowski product ``a . eta b`` over the component dim (-2), which is
    reduced away.

    Written on ``select`` rather than via :func:`_apply_metric`, which would materialize
    (and save for backward) a permuted copy of ``b``. Also not as ``a[..., 0, :]``: that
    indexing adds a no-op full ``slice`` of the last dim, whose ``slice_backward`` Inductor
    fused badly into the q/k/v-gradient layout kernel (+5 ms/step at batch 2048).
    """
    a0, a1, a2, a3 = (a.select(-2, i) for i in range(4))
    b0, b1, b2, b3 = (b.select(-2, i) for i in range(4))
    return a0 * b1 + a1 * b0 - a2 * b2 - a3 * b3


def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h_v = out[..., : hidden_v_channels * 4].unflatten(-1, (4, hidden_v_channels))
    h_s = out[..., hidden_v_channels * 4 :]

    h_v = _movedim(h_v, -4, -2).flatten(-2, -1)
    h_s = _movedim(h_s, -2, -3).flatten(-2, -1)
    return h_v, h_s


# ------------------------------------------------------------------------------------
# Attention backends (ported from the lgatr/lloca ``attention_backends`` packages and
# tagging-guide ``experiments/misc.py``): the backend is selected dynamically from the
# attention kwargs, so the same network code serves the dense zero-padded layout (SDPA
# with an ``attn_mask``) and the packed variable-length layout (flash-attention varlen
# kernels with ``cu_seqlens``, or xformers memory-efficient attention with a
# block-diagonal ``attn_bias``).
# ------------------------------------------------------------------------------------

ATTENTION_BACKENDS = ("auto", "native", "varlen", "flash", "xformers", "flex", "efficient")
# how the light-cone axis of an event is chosen; see LGATrSlimTagger's lightcone_frame
LIGHTCONE_FRAMES = ("sum", "principal", "collinear")

# kwargs that select the varlen kernels, following the upstream argument naming of
# torch.nn.attention.varlen.varlen_attn and flash_attn.flash_attn_varlen_func;
# ``attn_bias`` selects xformers.ops.memory_efficient_attention
_VARLEN_KWARGS = ("cu_seq_q", "cu_seq_k", "max_q", "max_k")
_FLASH_KWARGS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
_XFORMERS_KWARGS = ("attn_bias",)
_FLEX_KWARGS = ("block_mask",)
_EFFICIENT_KWARGS = ("cu_seqlens", "max_seqlen")

_flash_attn_varlen_func = None
_xformers_ops = None
_flex_ops = None
_compiled_flex_attention = None
_flex_block_sizes: dict[tuple, tuple[int, int]] = {}
_block_mask_builders: dict[tuple[int, int], Any] = {}


def _get_xformers_ops():
    """Import xformers' attention op and block-diagonal mask type."""
    global _xformers_ops
    if _xformers_ops is None:
        try:
            from xformers.ops import memory_efficient_attention
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        except ImportError as err:
            raise ImportError(
                "attention_backend='xformers' requires the xformers package "
                "(pip install xformers)."
            ) from err
        _xformers_ops = (memory_efficient_attention, BlockDiagonalMask)
    return _xformers_ops


def _get_flash_attn_varlen_func():
    """Import flash-attn's varlen kernel, preferring the FlashAttention-3 interface."""
    global _flash_attn_varlen_func
    if _flash_attn_varlen_func is None:
        try:
            # FlashAttention-3 (Hopper); falls back to the FlashAttention-2 package
            from flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            try:
                from flash_attn import flash_attn_varlen_func
            except ImportError as err:
                raise ImportError(
                    "attention_backend='flash' requires the flash-attn package "
                    "(pip install flash-attn, or flash-attn-3 on Hopper GPUs)."
                ) from err
        _flash_attn_varlen_func = flash_attn_varlen_func
    return _flash_attn_varlen_func


_HALF_CAST_WARNED = False
# the AMP settings already warned about (see LGATrSlimTagger._check_amp), once per process
_AMP_WARNED: set[str] = set()


def _half_cast_dtype(device: torch.device) -> torch.dtype:
    """Half-precision dtype to run a packed (varlen / flash / xformers) kernel in.

    Those kernels have no fp32 path, so fp32 inputs must be cast. Inside an autocast
    region the surrounding autocast dtype is used -- and in practice the inputs already
    arrive in it, so no cast happens at all. Outside one, fp16 is used: it keeps three
    more mantissa bits than bf16, which measurably matters here (see below).

    Note that this makes the packed backends *less accurate than* ``native`` and
    ``flex`` whenever those run in fp32, i.e. whenever the trainer is not under AMP. On a
    6-epoch JetClass run of the tag-slim config, val accuracy was 0.818 for fp32 attention
    against 0.808 (fp16) and 0.806 (bf16) -- and that ~1 pp gap comes from the precision,
    not from the packed layout: forcing the *dense* path to fp16 reproduced it, while the
    packed layout at matched fp16 precision cost <=0.3 pp (at the ~0.1-0.3 pp run-to-run
    noise floor). bf16 was tried here and is worse: fp16 does underflow the backward pass
    (the attention output gradients sit near 1e-7, below fp16's smallest subnormal, so
    most gradient *entries* flush to zero), but those entries are the smallest ones and
    carry ~1% of the gradient norm, whereas bf16's coarser mantissa degrades the forward
    values that matter -- worth ~1 pp of val accuracy in the packed layout.
    """
    global _HALF_CAST_WARNED

    if _autocast_active(device.type):
        dtype = torch.get_autocast_dtype(device.type)
        if dtype in (torch.float16, torch.bfloat16):
            return dtype

    if not _HALF_CAST_WARNED:
        _HALF_CAST_WARNED = True
        _logger.warning(
            "The packed attention backends have no fp32 kernel, so attention will run in "
            "float16 while the rest of the network stays in float32. This costs about 1% "
            "absolute accuracy against attention_backend='native', which keeps attention "
            "in float32; use 'native' if that matters more than the speedup."
        )
    return torch.float16


def _run_varlen_kernel(
    kernel, query, key, value, kernel_kwargs: dict, scale_kwarg: str = "scale"
) -> torch.Tensor:
    """Run a flash-attention varlen kernel on (1, heads, tokens, channels) tensors.

    The kernels expect packed (tokens, heads, channels) inputs in fp16/bf16; fp32 inputs
    are cast to :func:`_half_cast_dtype` and back (flash attention only supports half
    precision), and the head dim is zero-padded to a multiple of 8 (zero padding leaves
    the attention logits and the retained output channels unchanged; the softmax scale
    is passed explicitly via ``scale_kwarg`` so it keeps the un-padded 1/sqrt(channels)).
    """
    assert query.dim() == 4 and query.shape[0] == 1, (
        "varlen attention requires the packed (batch=1, heads, tokens, channels) layout."
    )

    in_dtype = None
    if query.dtype not in (torch.float16, torch.bfloat16):
        in_dtype = query.dtype
        dtype = _half_cast_dtype(query.device)
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)

    def reshape(x):
        # (1, heads, tokens, channels) -> (tokens, heads, channels)
        return x.squeeze(0).transpose(0, 1).contiguous()

    query, key, value = reshape(query), reshape(key), reshape(value)
    channels = query.shape[-1]
    pad = (-channels) % 8
    if pad:
        query, key, value = (F.pad(t, (0, pad)) for t in (query, key, value))

    out = kernel(query, key, value, **kernel_kwargs, **{scale_kwarg: channels**-0.5})
    if isinstance(out, tuple):
        # some flash-attn versions also return the logsumexp
        out = out[0]
    if pad:
        out = out[..., :channels]
    out = out.transpose(0, 1).unsqueeze(0).contiguous()

    if in_dtype is not None:
        out = out.to(in_dtype)
    return out


@torch.compiler.disable()
def _flash_attention(
    query, key, value, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None
):
    """Block-diagonal attention via flash-attn's ``flash_attn_varlen_func``."""
    kernel = _get_flash_attn_varlen_func()
    kernel_kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return _run_varlen_kernel(kernel, query, key, value, kernel_kwargs, scale_kwarg="softmax_scale")


def _has_varlen_attn() -> bool:
    """Whether this torch has ``torch.nn.attention.varlen.varlen_attn`` (torch >= 2.10)."""
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _varlen_attention(query, key, value, cu_seq_q=None, cu_seq_k=None, max_q=None, max_k=None):
    """Block-diagonal attention via torch's native ``varlen_attn`` (torch >= 2.10)."""
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as err:
        raise ImportError(
            "attention_backend='varlen' requires torch>=2.10 "
            "(torch.nn.attention.varlen.varlen_attn)."
        ) from err
    kernel_kwargs = dict(cu_seq_q=cu_seq_q, cu_seq_k=cu_seq_k, max_q=max_q, max_k=max_k)
    return _run_varlen_kernel(varlen_attn, query, key, value, kernel_kwargs, scale_kwarg="scale")


@torch.compiler.disable()
def _xformers_attention(query, key, value, attn_bias=None):
    """Block-diagonal attention via ``xformers.ops.memory_efficient_attention``.

    The inputs are (batch, heads, items, channels), as for SDPA; xformers expects
    (batch, items, heads, channels). As for the varlen kernels, fp32 inputs are cast
    to :func:`_half_cast_dtype` and back (xformers' fp32 cutlass kernel is unavailable on
    recent GPU architectures, e.g. sm_120, where only the fp16/bf16 flash-attention
    kernels remain), and the head dim is zero-padded to a multiple of 8 with the
    softmax scale pinned to the un-padded 1/sqrt(channels).
    """
    memory_efficient_attention, _ = _get_xformers_ops()

    in_dtype = None
    if query.dtype not in (torch.float16, torch.bfloat16):
        in_dtype = query.dtype
        dtype = _half_cast_dtype(query.device)
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)

    channels = query.shape[-1]
    pad = (-channels) % 8
    if pad:
        query, key, value = (F.pad(t, (0, pad)) for t in (query, key, value))

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    out = memory_efficient_attention(
        query, key, value, attn_bias=attn_bias, scale=channels**-0.5
    )
    out = out.transpose(1, 2).contiguous()

    if pad:
        out = out[..., :channels]
    if in_dtype is not None:
        out = out.to(in_dtype)
    return out


def _get_flex_ops():
    """Import ``flex_attention`` and ``create_block_mask`` (torch >= 2.5), uncompiled.

    Both are returned raw: see :func:`_flex_attention` for which of the two forms of
    ``flex_attention`` is used where, and :func:`_get_block_mask_builder` for how
    ``create_block_mask`` is compiled.
    """
    global _flex_ops
    if _flex_ops is None:
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        except ImportError as err:
            raise ImportError(
                "attention_backend='flex' requires torch>=2.5 "
                "(torch.nn.attention.flex_attention)."
            ) from err
        _flex_ops = (flex_attention, create_block_mask)
    return _flex_ops


def _get_block_mask_builder(block_size: tuple[int, int]):
    """A ``torch.compile``d ``create_block_mask``, one builder cached per block size.

    Compiling it is not optional: uncompiled, ``create_block_mask`` materializes a dense
    ``(tokens, tokens)`` mask before reducing it to blocks -- the O(tokens^2) allocation
    the packed layout exists to avoid.

    The compile has to make the token count symbolic (it changes every batch) while
    keeping the block size constant. The mask's shape is ``ceil(tokens / block)`` squared,
    and a symbolic block makes that a ratio of two unknowns, which Inductor cannot factor
    and reports as ``AssertionError: failed to set ranges``. Two things are needed to
    split them apart, because ``dynamic=True`` would unspecialize both:

    - the block size is closed over, not passed, and each builder is cached under it;
    - the token count comes from ``batch.shape[0]``, a tensor dim the caller marks
      dynamic (see :func:`_build_flex_block_mask`), so Dynamo never has to guess.

    Explicit marking rather than automatic dynamic: automatic dynamic responds to guards
    that keep failing, and after enough distinct token counts it unspecializes the block
    size too, so the failure would reappear a few steps into training.
    """
    builder = _block_mask_builders.get(block_size)
    if builder is not None:
        return builder

    _, create_block_mask = _get_flex_ops()

    def build(batch: torch.Tensor):
        def jagged_mask(b, h, q_idx, kv_idx):
            # tokens attend only within their own event
            return batch[q_idx] == batch[kv_idx]

        num_tokens = batch.shape[0]
        return create_block_mask(
            jagged_mask, None, None, num_tokens, num_tokens, device=batch.device,
            BLOCK_SIZE=block_size,
        )

    builder = torch.compile(build)
    _block_mask_builders[block_size] = builder
    return builder


def _get_compiled_flex_attention():
    """``torch.compile``d ``flex_attention``, for calls that are *not* already compiled.

    flex_attention only becomes a fused kernel once Inductor lowers its HOP; called from
    plain eager it falls back to an unfused implementation that materializes the full
    scores matrix, which the packed layout cannot afford. So the eager entry point
    (weaver without ``--compile``, ONNX export, ``--predict``) compiles it here, with
    ``dynamic=True`` for the batch-to-batch token count.
    """
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        flex_fn, _ = _get_flex_ops()
        _compiled_flex_attention = torch.compile(flex_fn, dynamic=True)
    return _compiled_flex_attention


def _flex_block_size(
    head_dim: int, device_type: str, dtypes: tuple[torch.dtype, ...]
) -> tuple[int, int]:
    """``(Q, KV)`` BlockMask tile that Inductor's flex_attention kernel will accept.

    The tile must be a multiple of the ``BLOCK_M`` / ``BLOCK_N`` of the Triton config
    Inductor picks, or the lowering fails with "Q and KV block size must be divisible by
    BLOCK_M and BLOCK_N". Without ``max_autotune`` there is only one candidate config, so
    a mismatch is fatal rather than just one lost choice.

    That config comes from a table keyed on ``(device capability, dtype, head_dim)`` with
    entries only at head_dim 64 / 128 / 192 / 256; any other head_dim gets a (64, 64)
    default. So the round head dims are the awkward ones -- on sm_120, fp32 head_dim 64
    asks for ``BLOCK_M=128``, which a fixed 64-token tile cannot satisfy.

    Hence: read the config Inductor will use and mirror its tiling, which divides by
    construction. Two alternatives are worse. A tile of 128, large enough for every table
    entry, costs 33% of a compiled fwd+bwd step on the default config (45 -> 60 ms, batch
    256, RTX 5090), because a coarser tile means more masked-out work. A square tile
    wastes the asymmetry: at head_dim 64, (128, 32) beats (128, 128) by 19%.

    ``dtypes`` is the precisions attention may run in, since the table is dtype-keyed and
    disagrees -- head_dim 28 on sm_120 wants (64, 64) in fp32 but (128, 128) in bf16.
    Attention follows the trainer's AMP setting, so this is the autocast dtype under AMP
    and ``(float32,)`` otherwise.

    This reads ``torch._inductor``, which is private, so a breakage falls back to
    (128, 128) -- coarse, but valid for every table shipped so far.
    """
    key = (head_dim, device_type, dtypes)
    cached = _flex_block_sizes.get(key)
    if cached is not None:
        return cached

    try:
        from torch._inductor.virtualized import V

        q_blocks, kv_blocks = [], []
        for dtype in dtypes:
            fwd = V.choices.get_flex_attention_fwd_configs(head_dim, dtype, device_type)
            bwd = V.choices.get_flex_attention_bwd_configs(head_dim, dtype, device_type)
            # With max_autotune these lists hold several configs, and ones that do not
            # divide are simply skipped -- so matching any single config is enough. Pick
            # the finest tiling. Without max_autotune each list holds one config anyway.
            f = min(fwd, key=lambda c: c.block_m * c.block_n)
            b = min(bwd, key=lambda c: max(c.block_m1, c.block_m2) * max(c.block_n1, c.block_n2))
            # the backward kernel has two tile pairs, (m1, n1) and (m2, n2), and checks
            # the mask against both
            q_blocks.append(math.lcm(f.block_m, b.block_m1, b.block_m2))
            kv_blocks.append(math.lcm(f.block_n, b.block_n1, b.block_n2))
        block_size = (math.lcm(*q_blocks), math.lcm(*kv_blocks))
    except Exception as err:  # noqa: BLE001 -- private API, any breakage falls back
        _logger.warning(
            "Could not read Inductor's flex_attention config for head_dim=%d (%s: %s); "
            "falling back to a 128x128 block-mask tile, which is valid but coarser than "
            "necessary for most head dims.",
            head_dim,
            type(err).__name__,
            err,
        )
        block_size = (128, 128)

    _flex_block_sizes[key] = block_size
    return block_size


@torch.compiler.disable()
def _build_flex_block_mask(batch: torch.Tensor, block_size: tuple[int, int]):
    """Build the block-diagonal ``BlockMask`` for the packed layout.

    Hidden from the *enclosing* torch.compile (weaver's ``--compile`` wraps the whole
    model): the mask's shapes are derived symbols like ``(tokens + k) // block_size``,
    and Inductor's ``statically_known_multiple_of`` oracle cannot factor those, so they
    reach the scheduler and raise ``CantSplit``. ``create_block_mask`` is compiled on its
    own inside :func:`_get_block_mask_builder`, so nothing is lost by keeping it out of
    the graph.
    """
    # the token count is the only thing that varies per batch; the block size must stay
    # a compile-time constant (see _get_block_mask_builder)
    torch._dynamo.mark_dynamic(batch, 0)
    return _get_block_mask_builder(block_size)(batch)


def _flex_attention(query, key, value, block_mask=None):
    """Block-diagonal attention via torch's ``flex_attention``.

    Unlike the varlen / flash / xformers kernels, flex_attention has an fp32 path, so
    this backend drops the padding without forcing half precision: it runs in whatever
    dtype its inputs carry, which is the trainer's AMP dtype under ``--use-amp`` and
    float32 otherwise. In Cartesian coordinates that fp32 path is worth ~1 pp of val
    accuracy over the half-precision packed backends (see :func:`_half_cast_dtype`); in
    the default light-cone coordinates half precision costs nothing, which is the whole
    point of the frame (see :func:`get_lightcone_frame`).

    The block mask makes this O(tokens x block) rather than the O(tokens^2) of a
    materialized block-diagonal SDPA mask, which is why the fp32 packed layout is
    tractable here at all.

    Under an enclosing ``torch.compile`` (weaver's ``--compile``) this calls the plain
    ``flex_attention``: Dynamo turns it into the ``flex_attention`` HOP and that same
    graph's Inductor lowering fuses it, so pre-wrapping it in a second ``torch.compile``
    would only nest one compile inside another. Outside a compiled region there is no
    such graph, so the separately compiled form is used instead.
    """
    if torch.compiler.is_compiling():
        flex_fn, _ = _get_flex_ops()
    else:
        flex_fn = _get_compiled_flex_attention()
    return flex_fn(query, key, value, block_mask=block_mask, scale=query.shape[-1] ** -0.5)


@torch.compiler.disable()
def _efficient_attention(query, key, value, cu_seqlens=None, max_seqlen=None):
    """Block-diagonal attention via torch's memory-efficient (cutlass) varlen kernel.

    The kernel ``F.scaled_dot_product_attention`` dispatches to for fp32 ``torch.nested``
    inputs, called directly on the packed layout: like ``flex`` it runs in the dtype of
    its inputs, so it has an exact fp32 path, but without a block mask it is ~2-3x faster
    than ``flex`` in fp32 (1.2 vs 3.4 ms fwd+bwd on 42k jet tokens, 6.1 vs 13.6 ms on
    121k event tokens, RTX 5090). Inductor cannot lower the op, hence the graph break.
    """
    in_shape = query.shape
    # (1, heads, tokens, channels) -> (1, tokens, heads, channels)
    query, key, value = (t.transpose(1, 2).contiguous() for t in (query, key, value))
    channels = query.shape[-1]
    pad = (-channels) % 8
    if pad:
        query, key, value = (F.pad(t, (0, pad)) for t in (query, key, value))
    out = torch.ops.aten._efficient_attention_forward(
        query, key, value, None, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        0.0, 0, torch.is_grad_enabled(), scale=channels**-0.5,
    )[0]
    if pad:
        out = out[..., :channels]
    return out.transpose(1, 2).reshape(in_shape)


def scaled_dot_product_attention(query, key, value, **attn_kwargs):
    """Attention with the backend selected by the given kwargs.

    No precision pinning of its own: every backend runs in the dtype of its inputs, so
    attention follows the trainer's AMP setting like the rest of the network.
    """
    if any(attn_kwargs.get(key_) is not None for key_ in _FLASH_KWARGS):
        return _flash_attention(query, key, value, **attn_kwargs)
    if any(attn_kwargs.get(key_) is not None for key_ in _VARLEN_KWARGS):
        return _varlen_attention(query, key, value, **attn_kwargs)
    if any(attn_kwargs.get(key_) is not None for key_ in _XFORMERS_KWARGS):
        return _xformers_attention(query, key, value, **attn_kwargs)
    if any(attn_kwargs.get(key_) is not None for key_ in _FLEX_KWARGS):
        return _flex_attention(query, key, value, **attn_kwargs)
    if any(attn_kwargs.get(key_) is not None for key_ in _EFFICIENT_KWARGS):
        return _efficient_attention(query, key, value, **attn_kwargs)
    return F.scaled_dot_product_attention(query, key, value, **attn_kwargs)


# ------------------------------------------------------------------------------------
# Packed (sparse) sequence utilities (ported from tagging-guide
# ``experiments/tagging/embedding.py`` and ``experiments/misc.py`` / lloca utils)
# ------------------------------------------------------------------------------------


def get_batch_from_ptr(ptr: torch.Tensor, num_items: int | None = None) -> torch.Tensor:
    """Per-token event indices from the (B+1,) cumulative-lengths pointer.

    Passing ``num_items`` (the total token count) avoids a device-to-host sync.
    """
    counts = ptr[1:] - ptr[:-1]
    return torch.arange(ptr.numel() - 1, device=ptr.device).repeat_interleave(
        counts, output_size=num_items
    )


def dense_to_sparse(dense_tensors: list[torch.Tensor], mask: torch.Tensor):
    """Pack dense zero-padded (B, P, ...) tensors into packed (tokens, ...) layout.

    Returns the packed tensors plus the per-token event index ``batch`` (tokens,) and the
    cumulative-lengths pointer ``ptr`` (B+1,), i.e. the ``cu_seqlens`` of the packed layout.
    """
    num_particles = mask.sum(dim=-1)
    ptr = torch.zeros(len(num_particles) + 1, device=mask.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(num_particles, dim=0)
    idxs = mask.flatten().nonzero().squeeze(-1)
    batch = get_batch_from_ptr(ptr, num_items=idxs.shape[0])

    sparse_tensors = [t.flatten(0, 1).index_select(0, idxs) for t in dense_tensors]
    return sparse_tensors, batch, ptr


def insert_global_tokens(ptr: torch.Tensor, batch: torch.Tensor, num_tokens: int):
    """Index bookkeeping for prepending one global token per event in the packed layout.

    Returns ``(global_idxs, nonglobal_idxs, ptr, batch, num_total)``: the positions of the
    global tokens and of the original tokens in the extended packed layout, plus the
    updated pointer / event indices for the ``num_total = num_tokens + B`` tokens.
    """
    batchsize = ptr.numel() - 1
    num_total = num_tokens + batchsize
    global_idxs = ptr[:-1] + torch.arange(batchsize, device=ptr.device)
    nonglobal_idxs = torch.arange(num_tokens, device=ptr.device) + batch + 1
    ptr = ptr + torch.arange(batchsize + 1, device=ptr.device)
    batch = get_batch_from_ptr(ptr, num_items=num_total)
    return global_idxs, nonglobal_idxs, ptr, batch, num_total


def get_sparse_attention_kwargs(
    ptr: torch.Tensor,
    batch: torch.Tensor,
    maxlen: int,
    attention_backend: str,
    flex_head_dim: int | None = None,
    flex_dtypes: tuple[torch.dtype, ...] = (torch.float32,),
) -> dict:
    """Attention kwargs for block-diagonal attention over the packed layout.

    On CUDA this returns the ``cu_seqlens`` kwargs of the requested varlen kernel
    (``maxlen`` may be any upper bound on the sequence lengths, so no device-to-host
    sync is needed), the ``attn_bias`` kwarg for the xformers backend (whose
    ``BlockDiagonalMask`` is built from host-side sequence lengths, so this backend
    incurs a device-to-host sync), or the ``block_mask`` kwarg for the flex backend
    (built on device from ``batch``, no sync). On CPU, where the fused attention kernels are
    unavailable, it falls back to a materialized block-diagonal SDPA mask (as in
    tagging-guide).

    ``flex_head_dim`` (the model's per-head attention dimension) and ``flex_dtypes`` (the
    precisions attention may run in) are required by the flex backend only, to size its
    block mask -- see :func:`_flex_block_size`.
    """
    if ptr.device.type == "cpu":
        attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        return {"attn_mask": attn_mask}
    if attention_backend == "xformers":
        _, BlockDiagonalMask = _get_xformers_ops()
        seqlens = (ptr[1:] - ptr[:-1]).tolist()
        return {"attn_bias": BlockDiagonalMask.from_seqlens(seqlens)}
    if attention_backend == "flex":
        # The tile wants to be small -- flex's 128-token default is wider than a whole
        # jet (JetClass averages ~70 constituents), so most of each diagonal tile would
        # be masked-out work; halving it cut a compiled fwd+bwd step from 90 ms to 70 ms
        # at batch 512 on an RTX 5090. But it cannot be pinned small either: it has to
        # divide the Triton config Inductor picks. _flex_block_size settles that.
        if flex_head_dim is None:
            raise ValueError(
                "the flex backend needs flex_head_dim to size the block mask; "
                "pass the per-head attention dimension of the model."
            )
        block_size = _flex_block_size(flex_head_dim, batch.device.type, flex_dtypes)
        return {"block_mask": _build_flex_block_mask(batch, block_size=block_size)}
    cu_seqlens = ptr.to(torch.int32)
    if attention_backend == "efficient":
        return {"cu_seqlens": cu_seqlens, "max_seqlen": maxlen}
    if attention_backend == "flash":
        return {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": maxlen,
            "max_seqlen_k": maxlen,
        }
    elif attention_backend == "varlen":
        return {
            "cu_seq_q": cu_seqlens,
            "cu_seq_k": cu_seqlens,
            "max_q": maxlen,
            "max_k": maxlen,
        }
    raise ValueError(
        f"Unsupported attention backend for the packed layout: {attention_backend}. "
        f"Supported backends: {ATTENTION_BACKENDS}."
    )


def _freeze_dead_tail(
    norm: nn.Module, mlp: nn.Module, out_v_channels: int, out_s_channels: int
) -> None:
    """Freeze last-block params that cannot receive grads when an output stream is empty."""
    if out_v_channels == 0:
        if norm.weight_v is not None:
            norm.weight_v.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if name.endswith("weight_v") or name.endswith("gate_scale"):
                p.requires_grad_(False)
    if out_s_channels == 0:
        if norm.weight_s is not None:
            norm.weight_s.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if "linear_s" in name:
                p.requires_grad_(False)


class SlimDropout(nn.Module):
    """Dropout for vector and scalar features.

    For vector features the same dropout mask is applied to all four components of each vector.
    """

    def __init__(self, dropout_prob: float) -> None:
        super().__init__()
        self._dropout_prob = dropout_prob

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., 4, v_channels); scalars: (..., s_channels)
        if not self.training or self._dropout_prob == 0.0:
            return vectors, scalars

        # have to reshape vectors because dropout1d constrains input shape
        flat_v = vectors.transpose(-1, -2).reshape(-1, 4)
        outputs_v = (
            F.dropout1d(flat_v, p=self._dropout_prob, training=True)
            .reshape(*vectors.shape[:-2], vectors.shape[-1], 4)
            .transpose(-1, -2)
        )
        outputs_s = F.dropout(scalars, p=self._dropout_prob, training=True)
        return outputs_v, outputs_s


class SlimRMSNorm(nn.Module):
    """Joint RMS normalization over vector and scalar features.

    For vectors the absolute value of the squared norm is used; otherwise the squared norm could
    be negative under the Lorentz metric.
    """

    _lightcone = False  # set by LGATrSlim(lightcone=...)

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        epsilon: float = 0.01,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)
        if elementwise_affine:
            self.weight_v = nn.Parameter(torch.ones(v_channels))
            self.weight_s = nn.Parameter(torch.ones(s_channels))
            # zero-size params get grads only sometimes under compile, breaking DDP
            for weight in (self.weight_v, self.weight_s):
                if weight.numel() == 0:
                    weight.requires_grad_(False)
        else:
            self.register_parameter("weight_v", None)
            self.register_parameter("weight_s", None)

    @minimum_autocast_precision(torch.float32, output="high")
    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., 4, v_channels); scalars: (..., s_channels)
        if self._lightcone:
            v_squared_norm = _lightcone_product(vectors, vectors).abs()
        else:
            v_squared_norm = (vectors.square() * self.metric[..., None]).sum(-2).abs()
        s_squared_norm = scalars.square()
        total_features = v_squared_norm.shape[-1] + s_squared_norm.shape[-1]
        mean_squared_norms = (v_squared_norm.sum(-1) + s_squared_norm.sum(-1)) / total_features
        norm = torch.rsqrt(mean_squared_norms + self.epsilon)

        outputs_v = vectors * norm[..., None, None]
        outputs_s = scalars * norm[..., None]
        if self.elementwise_affine:
            outputs_v = outputs_v * self.weight_v
            outputs_s = outputs_s * self.weight_s
        return outputs_v, outputs_s


class SlimLinear(nn.Module):
    """Linear layer for vector and scalar features.

    The vector and scalar streams are kept separate; mixing happens elsewhere.
    """

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()
        self._in_v_channels = in_v_channels
        self._out_v_channels = out_v_channels
        self._in_s_channels = in_s_channels
        self._out_s_channels = out_s_channels
        self._bias = bias

        self.weight_v = nn.Parameter(torch.empty((out_v_channels, in_v_channels)))
        self.linear_s: nn.Linear | None
        if in_s_channels and out_s_channels:
            self.linear_s = nn.Linear(in_s_channels, out_s_channels, bias=bias)
        else:
            self.linear_s = None

        self.reset_parameters(initialization)

        # zero-size params get grads only sometimes under compile, breaking DDP
        if self.weight_v.numel() == 0:
            self.weight_v.requires_grad_(False)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., 4, in_v_channels) -> (..., 4, out_v_channels)
        # scalars: (..., in_s_channels) -> (..., out_s_channels)
        # Both GEMMs follow the trainer's AMP setting: the vector stream carries no
        # metric here (that is the norms' and the gates' business, which stay fp32), so
        # there is nothing for half precision to cancel.
        outputs_v = F.linear(vectors, self.weight_v)
        if self.linear_s is not None:
            outputs_s = self.linear_s(scalars)
        else:
            outputs_s = scalars.new_zeros(*scalars.shape[:-1], self._out_s_channels)
        return outputs_v, outputs_s

    def reset_parameters(self, initialization: str, additional_factor: float = 1.0) -> None:
        """Re-initialize the weights with the given scheme."""
        if initialization == "default":
            v_factor = additional_factor
            s_factor = additional_factor
        elif initialization == "small":
            v_factor = 0.1 * additional_factor
            s_factor = 0.1 * additional_factor
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        if self.weight_v.numel() > 0:
            fan_in = max(self._in_v_channels, 1)
            bound = v_factor / math.sqrt(fan_in)
            nn.init.uniform_(self.weight_v, a=-bound, b=bound)

        if self.linear_s is not None:
            fan_in = max(self._in_s_channels, 1)
            bound = s_factor / math.sqrt(fan_in)
            nn.init.uniform_(self.linear_s.weight, a=-bound, b=bound)
            if self.linear_s.bias is not None:
                nn.init.zeros_(self.linear_s.bias)


class SlimGLU(nn.Module):
    """Gated linear unit (GLU) for vector and scalar features.

    Scalar gates are computed from scalar features; vector gates are computed from inner products
    of (transformed) vector features.
    """

    _lightcone = False  # set by LGATrSlim(lightcone=...)

    def __init__(
        self,
        in_v_channels: int,
        out_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.linear = SlimLinear(
            in_v_channels=in_v_channels,
            out_v_channels=3 * out_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=2 * out_s_channels,
        )
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.nonlinearity_v = (
            get_nonlinearity(nonlinearity_v) if nonlinearity_v is not None else self.nonlinearity
        )
        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)
        # set by LGATrSlim(gate_norm="euclidean"): per-channel scale of the scale-free gate
        self.gate_scale: nn.Parameter | None = None

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., 4, in_v_channels) -> (..., 4, out_v_channels)
        # scalars: (..., in_s_channels) -> (..., out_s_channels)
        v_full, s_full = self.linear(vectors, scalars)
        v_pre, v_gates_1, v_gates_2 = v_full.chunk(3, dim=-1)
        s_pre, s_gates = s_full.chunk(2, dim=-1)

        v_gates = self._get_inner_product(v_gates_1, v_gates_2)

        outputs_v = self.nonlinearity_v(v_gates) * v_pre
        outputs_s = self.nonlinearity(s_gates) * s_pre
        return outputs_v, outputs_s

    @minimum_autocast_precision(torch.float32)
    def _get_inner_product(self, v_gates_1: torch.Tensor, v_gates_2: torch.Tensor) -> torch.Tensor:
        if self._lightcone:
            product = _lightcone_product(v_gates_1, v_gates_2).unsqueeze(-2)
        else:
            product = ((v_gates_1 * v_gates_2) * self.metric[..., None]).sum(dim=-2, keepdim=True)
        if self.gate_scale is not None:
            # scale-free gate: |<a, b>| <= ||a|| ||b|| (Euclidean, Cauchy-Schwarz), so the
            # argument is a learned per-channel scale times a number in [-1, 1] and the gate
            # cannot saturate just because the vector stream grew. The Euclidean norms are
            # frame dependent, so this gate is not a Lorentz scalar: it breaks the Lorentz
            # equivariance of the block (rotations are still respected in Cartesian coordinates)
            norms = v_gates_1.square().sum(-2, keepdim=True).sqrt() * v_gates_2.square().sum(-2, keepdim=True).sqrt()
            return self.gate_scale * product / (norms + 1e-6)
        # 0.5 = 1/sqrt(4) controls the scale, like 1/sqrt(d_k) in attention
        return 0.5 * product


class SlimSelfAttention(nn.Module):
    """Self-attention for Lorentz vectors and scalar features."""

    _lightcone = False  # set by LGATrSlim(lightcone=...): selects the light-cone metric

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        attn_ratio: int = 1,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_v_channels = max(attn_ratio * v_channels // num_heads, 1)
        self.hidden_s_channels = max(attn_ratio * s_channels // num_heads, 4)
        self.num_heads = num_heads

        self.register_buffer("metric", torch.tensor([1.0, -1.0, -1.0, -1.0]), persistent=False)

        self.linear_in = SlimLinear(
            in_v_channels=v_channels,
            out_v_channels=3 * self.hidden_v_channels * self.num_heads,
            in_s_channels=s_channels,
            out_s_channels=3 * self.hidden_s_channels * self.num_heads,
            bias=False,
            initialization="small",
        )
        self.linear_out = SlimLinear(
            in_v_channels=self.hidden_v_channels * self.num_heads,
            out_v_channels=v_channels,
            in_s_channels=self.hidden_s_channels * self.num_heads,
            out_s_channels=s_channels,
            initialization="small",
        )
        self.norm = SlimRMSNorm(
            self.hidden_v_channels,
            self.hidden_s_channels,
            elementwise_affine=False,
        )
        if dropout_prob is not None:
            self.dropout = SlimDropout(dropout_prob)
        else:
            self.dropout = None

    def _pre_attention_reshape(
        self, qkv_v: torch.Tensor, qkv_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_v = qkv_v.unflatten(-1, (3, self.hidden_v_channels, self.num_heads))
        qkv_v = _movedim(_movedim(qkv_v, -3, 0), -1, -4)
        qkv_s = qkv_s.unflatten(-1, (3, self.hidden_s_channels, self.num_heads))
        qkv_s = _movedim(_movedim(qkv_s, -3, 0), -1, -3)

        # norm QK to avoid attention logit blowup (standard in LLMs)
        # we find that normalizing V as well helps with stability+performance
        qkv_v, qkv_s = self.norm(qkv_v, qkv_s)
        q_v, k_v, v_v = qkv_v.unbind(0)
        q_s, k_s, v_s = qkv_s.unbind(0)

        q_v = _apply_metric(q_v, self.metric, self._lightcone)

        q = torch.cat([q_v.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    @staticmethod
    def _attention_dtype(dtype: torch.dtype) -> torch.dtype:
        """The dtype attention runs in: the trainer's autocast dtype under AMP, else the
        inputs'.

        Taking the autocast dtype rather than inheriting the inputs' keeps this correct
        whichever way the fp32 islands that remain -- the norms and the gate contractions,
        which hold the metric -- happen to round their outputs.
        """
        if dtype.is_floating_point and _autocast_active():
            for device_type in ("cuda", "cpu"):
                if _autocast_active(device_type):
                    return torch.get_autocast_dtype(device_type)
        return dtype

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., items, 4, v_channels); scalars: (..., items, s_channels)
        qkv_v, qkv_s = self.linear_in(vectors, scalars)

        q, k, v = self._pre_attention_reshape(qkv_v, qkv_s)
        # Attention follows the trainer's AMP setting. The default light-cone coordinates
        # make half-precision attention as accurate as fp32 (see get_lightcone_frame); in
        # Cartesian ones it costs ~1 pp, so run those in fp32 (i.e. without --use-amp).
        dtype = self._attention_dtype(q.dtype)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        out = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        h_v, h_s = _post_attention_reshape(out, self.hidden_v_channels)

        outputs_v, outputs_s = self.linear_out(h_v, h_s)

        if self.dropout is not None:
            outputs_v, outputs_s = self.dropout(outputs_v, outputs_s)
        return outputs_v, outputs_s


class SlimMLP(nn.Module):
    """Multi-layer perceptron for vector and scalar features."""

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        num_layers: int = 2,
        dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, f"SlimMLP needs num_layers >= 2, got {num_layers}"
        layers: list[nn.Module] = []

        v_channels_list = [v_channels] + [mlp_ratio * v_channels] * (num_layers - 1) + [v_channels]
        s_channels_list = [s_channels] + [mlp_ratio * s_channels] * (num_layers - 1) + [s_channels]

        for i in range(num_layers - 1):
            layers.append(
                SlimGLU(
                    in_v_channels=v_channels_list[i],
                    out_v_channels=v_channels_list[i + 1],
                    in_s_channels=s_channels_list[i],
                    out_s_channels=s_channels_list[i + 1],
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                )
            )
            if dropout_prob is not None:
                layers.append(SlimDropout(dropout_prob))
        layers.append(
            SlimLinear(
                in_v_channels=v_channels_list[-2],
                out_v_channels=v_channels_list[-1],
                in_s_channels=s_channels_list[-2],
                out_s_channels=s_channels_list[-1],
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_v, h_s = vectors, scalars

        for layer in self.layers:
            h_v, h_s = layer(h_v, scalars=h_s)

        return h_v, h_s


class SlimBlock(nn.Module):
    """A single block of the L-GATr-slim network.

    Pre-norm + self-attention + residual, then pre-norm + MLP + residual.
    """

    def __init__(
        self,
        v_channels: int,
        s_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)
        self.norm2 = SlimRMSNorm(v_channels, s_channels, elementwise_affine=norm_elementwise_affine)

        self.attention = SlimSelfAttention(
            v_channels=v_channels,
            s_channels=s_channels,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            dropout_prob=dropout_prob,
        )

        self.mlp = SlimMLP(
            v_channels=v_channels,
            s_channels=s_channels,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., items, 4, v_channels); scalars: (..., items, s_channels)
        h_v, h_s = self.norm1(vectors, scalars)

        h_v, h_s = self.attention(h_v, h_s, **attn_kwargs)

        outputs_v = vectors + h_v
        outputs_s = scalars + h_s

        h_v, h_s = self.norm2(outputs_v, outputs_s)

        h_v, h_s = self.mlp(h_v, h_s)

        outputs_v = outputs_v + h_v
        outputs_s = outputs_s + h_s

        return outputs_v, outputs_s


# ------------------------------------------------------------------------------------
# L-GATr-slim network (ported from lgatr/nets/slim.py)
# ------------------------------------------------------------------------------------


class LGATrSlim(nn.Module):
    """L-GATr-slim network.

    A slimmer L-GATr variant that operates on Lorentz vectors and scalars (no full multivector
    representation). Stacks ``num_blocks`` :class:`SlimBlock` modules between initial and
    final :class:`SlimLinear` layers.

    Parameters
    ----------
    num_blocks
        Number of Lorentz-transformer blocks.
    in_v_channels / out_v_channels / hidden_v_channels
        Number of input/output/hidden vector channels.
    in_s_channels / out_s_channels / hidden_s_channels
        Number of input/output/hidden scalar channels.
    num_heads
        Number of attention heads.
    nonlinearity
        Nonlinearity for the MLP layers.
    nonlinearity_v
        Optional override for the vector-path gate nonlinearity in every GLU. ``None`` falls
        back to ``nonlinearity``.
    mlp_ratio / attn_ratio
        Expansion ratios for MLP / attention hidden channels.
    num_layers_mlp
        Number of layers in each MLP (must be ``>= 2``).
    dropout_prob
        Dropout probability.
    norm_elementwise_affine
        Whether the block :class:`SlimRMSNorm` instances learn a per-channel gain.
    checkpoint_blocks
        Whether to use gradient checkpointing for the blocks.
    naive_amp
        Whether to bypass the fp32 precision islands so the whole forward runs in the
        surrounding autocast dtype (e.g. bf16). When ``False`` (default), the norms and the
        gate contractions -- the two places that contract the metric -- stay fp32 under
        autocast; everything else (the GEMMs and attention) follows AMP.
    lightcone
        Whether the vector inputs are in the light-cone coordinates of
        :func:`get_lightcone_frame` rather than Cartesian ones. Every metric contraction
        then uses the light-cone metric.
    compile
        Whether to wrap the model's forward with :func:`torch.compile`.
    compile_kwargs
        Dict forwarded verbatim to :func:`torch.compile` when ``compile=True``.
    gate_norm / gate_scale_init
        ``"euclidean"`` makes the vector gate scale-free: its argument becomes a learned
        per-channel scale (initialised to ``gate_scale_init``) times ``<a, b> / (||a|| ||b||)``,
        which lies in [-1, 1]. ``"none"`` (default) keeps ``0.5 <a, b>``. The Euclidean norms
        depend on the frame, so ``"euclidean"`` breaks Lorentz equivariance.
    """

    def __init__(
        self,
        num_blocks: int,
        in_v_channels: int,
        out_v_channels: int,
        hidden_v_channels: int,
        in_s_channels: int,
        out_s_channels: int,
        hidden_s_channels: int,
        num_heads: int,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        mlp_ratio: int = 2,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        naive_amp: bool = False,
        lightcone: bool = False,
        compile: bool = False,
        compile_kwargs: dict | None = None,
        gate_norm: str = "none",
        gate_scale_init: float = 4.0,
    ) -> None:
        super().__init__()
        self._naive_amp = naive_amp
        if gate_norm not in ("none", "euclidean"):
            raise ValueError(f"Unsupported gate_norm: {gate_norm}. Supported: 'none', 'euclidean'.")

        self.linear_in = SlimLinear(
            in_v_channels=in_v_channels,
            in_s_channels=in_s_channels,
            out_v_channels=hidden_v_channels,
            out_s_channels=hidden_s_channels,
        )

        self.blocks = nn.ModuleList(
            [
                SlimBlock(
                    v_channels=hidden_v_channels,
                    s_channels=hidden_s_channels,
                    num_heads=num_heads,
                    nonlinearity=nonlinearity,
                    nonlinearity_v=nonlinearity_v,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    num_layers_mlp=num_layers_mlp,
                    dropout_prob=dropout_prob,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_blocks)
            ]
        )

        self.linear_out = SlimLinear(
            in_v_channels=hidden_v_channels,
            in_s_channels=hidden_s_channels,
            out_v_channels=out_v_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks
        # the coordinates decide the metric; precision is not a coordinate choice, it
        # follows the trainer's AMP setting everywhere except the fp32 islands that hold
        # the metric (the norms and the gate contractions, see minimum_autocast_precision)
        for m in self.modules():
            if isinstance(m, (SlimRMSNorm, SlimGLU, SlimSelfAttention)):
                m._lightcone = lightcone
            if isinstance(m, SlimGLU) and gate_norm == "euclidean":
                # no random draw: the other weights of a seed are unchanged
                out_v = m.linear._out_v_channels // 3
                m.gate_scale = nn.Parameter(torch.full((out_v,), float(gate_scale_init)))

        if num_blocks:
            _freeze_dead_tail(
                self.blocks[-1].norm2, self.blocks[-1].mlp, out_v_channels, out_s_channels
            )

        if compile:
            # rebind self.forward rather than patching the class to keep compilation instance-local
            self.forward = torch.compile(self.forward, **dict(compile_kwargs or {}))

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        vectors
            Lorentz vectors of shape ``(..., items, in_v_channels, 4)``.
        scalars
            Scalar features of shape ``(..., items, in_s_channels)``.
        **attn_kwargs
            Optional keyword arguments forwarded to attention (e.g. ``attn_mask``).

        Returns
        -------
        outputs_v
            Lorentz vectors of shape ``(..., items, out_v_channels, 4)``.
        outputs_s
            Scalar features of shape ``(..., items, out_s_channels)``.
        """
        _require_scalars(scalars=scalars)
        with naive_amp(self._naive_amp):
            return self._forward(vectors, scalars, **attn_kwargs)

    def _forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden layers keep vectors channel-last (..., 4, channels) so the vector linears run
        # as flat GEMMs; only the public interface uses (..., channels, 4)
        h_v, h_s = self.linear_in(vectors.transpose(-2, -1), scalars)

        for block in self.blocks:
            if self._checkpoint_blocks:
                h_v, h_s = checkpoint(block, h_v, h_s, use_reentrant=False, **attn_kwargs)
            else:
                h_v, h_s = block(h_v, h_s, **attn_kwargs)

        outputs_v, outputs_s = self.linear_out(h_v, h_s)
        return outputs_v.transpose(-2, -1), outputs_s


# ------------------------------------------------------------------------------------
# Jet-tagging wrapper (ported from tagging-guide experiments/tagging/{wrappers,embedding}.py)
# ------------------------------------------------------------------------------------


def get_spurion(
    beam_reference: str | None,
    add_time_reference: bool,
    two_beams: bool,
) -> torch.Tensor:
    """Construct the symmetry-breaking reference vectors ("spurions").

    Parameters
    ----------
    beam_reference
        ``"lightlike"``, ``"spacelike"``, ``"timelike"``, ``"all"``, or ``None``.
    add_time_reference
        Whether to add the time direction as a reference to the network.
    two_beams
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam.

    Returns
    -------
    spurion
        Tensor of shape ``(n_spurions, 4)`` in ``(E, px, py, pz)`` convention.
    """
    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, dtype=torch.float32).reshape(1, 4)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 3] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)
    elif beam_reference == "all":
        beam = torch.tensor(
            [
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
    elif beam_reference is None:
        beam = torch.empty(0, 4, dtype=torch.float32)
    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = torch.tensor([1, 0, 0, 0], dtype=torch.float32).reshape(1, 4)
    else:
        time = torch.empty(0, 4, dtype=torch.float32)

    return torch.cat((beam, time), dim=-2)


def get_lightcone_frame(fourmomenta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-event orthogonal map from Cartesian ``(E, px, py, pz)`` to light-cone coordinates.

    With ``n`` the jet direction (sum of the valid constituents), ``e1 = z x n / |z x n|``
    and ``e2 = n x e1``, the rows are ``x+ = (E + p.n)/sqrt2``, ``x- = (E - p.n)/sqrt2``,
    ``x1 = p.e1`` and ``x2 = p.e2``. The map ``T`` is orthogonal and ``T eta T^T`` is the
    light-cone metric of :func:`_apply_metric`, so with every vector (spurions included)
    transformed by ``T`` the network computes the same function in exact arithmetic.

    What changes is the conditioning. Jet constituents are nearly collinear with ``n``,
    so in Cartesian coordinates a Minkowski product ``E E' - p.p'`` is a small difference
    of two large numbers and rounding each component to half precision destroys it. Here
    the small ``E - p.n`` is formed once, in float64, and stored as a number of its own.

    Returns ``(N, 4, 4)`` float64.
    """
    momenta = fourmomenta.to(torch.float64)
    # masked with ``where`` rather than multiplied by the mask: 0 * nan is nan, and
    # padded slots are allowed to hold anything
    keep = mask.to(torch.bool).unsqueeze(-1)
    p = torch.where(keep, momenta, torch.zeros_like(momenta)).sum(dim=1)[..., 1:]
    return _lightcone_frame_from_sum(p)


def _lightcone_frame_from_sum(p: torch.Tensor) -> torch.Tensor:
    """The frame of :func:`get_lightcone_frame` from the summed momentum ``p`` ``(N, 3)``."""
    n = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nx, ny, nz = n.unbind(-1)
    rho = torch.sqrt(nx**2 + ny**2).clamp_min(1e-12)
    zero, s = torch.zeros_like(nx), torch.full_like(nx, 2**-0.5)
    rows = [
        torch.stack([s, s * nx, s * ny, s * nz], dim=-1),
        torch.stack([s, -s * nx, -s * ny, -s * nz], dim=-1),
        torch.stack([zero, -ny / rho, nx / rho, zero], dim=-1),
        torch.stack([zero, -nz * nx / rho, -nz * ny / rho, rho], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


# ------------------------------------------------------------------------------------
# Event-level light-cone axes
#
# get_lightcone_frame above assumes a jet: one axis, taken from the summed momentum. A
# full event has no such axis -- its momentum sum points along the beam and says nothing
# about where the particles are -- so the axis here is fitted to the particle directions
# instead. It is still one axis, hence one frame, per event.
#
# None of this is a physics selection: the axis is a numerical coordinate choice, and the
# network function is unchanged in exact arithmetic for any axis.
# ------------------------------------------------------------------------------------

# magnitudes below this are treated as zero when normalizing a direction
_AXIS_EPS = 1e-30


def _weighted_directions(
    fourmomenta: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights, directions)`` of the valid particles, as used by the axis fits.

    Weights are ``|p_i|`` normalized by their event sum; padded particles and
    particles with no spatial momentum get zero weight and a zero direction, so they
    never contribute to an axis -- masked with ``where``, not multiplied by zero, so
    that padding holding infinities or NaNs cannot poison the reduction.
    """
    momenta = fourmomenta[..., 1:]
    magnitude = momenta.norm(dim=-1)
    usable = mask.to(torch.bool) & (magnitude > 0)
    directions = torch.where(
        usable.unsqueeze(-1),
        momenta / magnitude.clamp_min(_AXIS_EPS).unsqueeze(-1),
        torch.zeros_like(momenta),
    )
    weights = torch.where(usable, magnitude, torch.zeros_like(magnitude))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(_AXIS_EPS), directions


def _direction_tensor(weights: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """``S = sum_i w_i n_i n_i^T``, ``(B, 3, 3)``.

    The uncentered (linearized sphericity) direction tensor: opposite directions reinforce
    each other, which is what a light-cone axis wants -- it protects both of its poles --
    whereas a centered covariance would cancel them.
    """
    return torch.einsum("bn,bni,bnj->bij", weights, directions, directions)


def leading_eigenvector(matrices: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the largest eigenvalue of symmetric PSD ``(..., 3, 3)`` inputs.

    The sign of an eigenvector is physically immaterial -- a light-cone frame about ``n``
    protects both poles -- but it is fixed deterministically (largest-magnitude component
    positive) so that frames are reproducible. Inputs that are (numerically) zero, i.e.
    events with no directional weight, fall back to the z axis.
    """
    trace = matrices.diagonal(dim1=-2, dim2=-1).sum(-1)
    usable = trace > _AXIS_EPS
    eye = torch.eye(3, dtype=matrices.dtype, device=matrices.device).expand_as(matrices)
    _, eigenvectors = torch.linalg.eigh(torch.where(usable[..., None, None], matrices, eye))
    axis = eigenvectors[..., -1]  # eigh returns ascending eigenvalues
    largest = axis.abs().argmax(dim=-1, keepdim=True)
    sign = torch.sign(axis.gather(-1, largest))
    axis = axis * torch.where(sign == 0, torch.ones_like(sign), sign)
    fallback = torch.zeros_like(axis)
    fallback[..., 2] = 1.0
    return torch.where(usable[..., None], axis, fallback)


def lightcone_frame_from_axis(axis: torch.Tensor) -> torch.Tensor:
    """Light-cone frame ``(..., 4, 4)`` about a unit spatial ``axis`` ``(..., 3)``.

    Rows ``(1, n)/sqrt2``, ``(1, -n)/sqrt2``, ``(0, e1)``, ``(0, e2)``, i.e.
    ``x+ = (E + p.n)/sqrt2``, ``x- = (E - p.n)/sqrt2``, ``x1 = p.e1``, ``x2 = p.e2``. For
    any orthonormal ``(n, e1, e2)`` the map is Euclidean-orthogonal (so its inverse is its
    transpose) and satisfies ``T^T J T = eta`` for the light-cone metric ``J`` of
    :func:`_apply_metric`, so the network computes the same function as in Cartesian
    coordinates -- see :func:`get_lightcone_frame` for why the conditioning differs.

    Unlike :func:`get_lightcone_frame`, which builds the transverse pair from ``z x n``
    and degenerates as ``n`` approaches the beam axis, ``e1`` is taken from the Cartesian
    unit vector *least* aligned with ``n``. That bounds ``|h x n| >= sqrt(2/3)``, so the
    construction is well conditioned for every direction -- which matters for events,
    whose axes are often close to the beam.
    """
    least_aligned = axis.abs().argmin(dim=-1, keepdim=True)
    helper = torch.zeros_like(axis).scatter_(-1, least_aligned, 1.0)
    e1 = torch.linalg.cross(helper, axis, dim=-1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(_AXIS_EPS)
    e2 = torch.linalg.cross(axis, e1, dim=-1)
    half = torch.full_like(axis[..., :1], 2**-0.5)
    zero = torch.zeros_like(half)
    rows = [
        torch.cat([half, half * axis], dim=-1),
        torch.cat([half, -half * axis], dim=-1),
        torch.cat([zero, e1], dim=-1),
        torch.cat([zero, e2], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def get_principal_axis(fourmomenta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Event principal axis, ``(B, 3)`` unit vectors.

    The leading eigenvector of the direction tensor ``sum_i w_i n_i n_i^T`` with
    ``w_i = |p_i|`` normalized by the event momentum sum. Particle-only -- no jets or
    external clustering -- and defined for any set of particles, including a balanced
    dijet whose momentum sum vanishes.
    """
    weights, directions = _weighted_directions(fourmomenta, mask)
    return leading_eigenvector(_direction_tensor(weights, directions))


def collinearity_weights(
    directions: torch.Tensor,
    weights: torch.Tensor,
    usable: torch.Tensor,
    scale: float,
    chunk: int = 256,
) -> torch.Tensor:
    """``c_i = w_i sum_{j != i} w_j / (theta_ij^2 + scale^2)``: how much *nearby* momentum ``i`` has.

    This is the weight that makes a single light-cone axis minimize the right thing. The
    relative half-precision error of a pair's Minkowski product is
    ``~ eps (theta_i^2 + theta_j^2) / theta_ij^2`` with ``theta`` measured from the axis,
    so the damage an axis ``a`` does to an event is

        D(a) = sum_{i<j} w_ij (theta_i^2 + theta_j^2) / theta_ij^2 = sum_i c_i sin^2 theta_i(a),

    i.e. the same eigenproblem :func:`get_principal_axis` solves, but weighted by ``c_i``
    instead of by the momentum ``w_i``. ``principal`` minimizes the momentum variance
    about the axis, which is not the same objective: an isolated hard particle pulls the
    axis towards itself even though none of its pairs are ill-conditioned, while a
    collimated cluster of soft particles -- where every internal pair *is*
    ill-conditioned -- barely counts.

    ``scale`` is the angular scale below which a pair counts as collinear; it also
    regulates ``theta_ij -> 0``, where the pair's product vanishes anyway.

    The pairwise kernel is the only O(N^2) term in the geometry, so it is computed in
    single precision (the axis it feeds is a coordinate choice, not a physical quantity)
    and in chunks over the batch, to keep a (B, N, N) intermediate off the peak.
    """
    reference = directions.dtype
    directions32 = directions.to(torch.float32)
    weights32 = weights.to(torch.float32)
    valid = usable.to(torch.float32)
    out = []
    # `.to(torch.float32)` is NOT enough: under the trainer's outer autocast, bmm casts
    # float32 operands down to the autocast dtype (float64 operands are left alone, which
    # is why the rest of the geometry is safe). In bfloat16 the spacing near cos = 1 is
    # 2^-8, so 1 - cos is unresolvable below ~0.09 rad and every collinear pair -- exactly
    # the ones this weight is about -- collapses to theta = 0 and saturates the kernel.
    with torch.autocast(device_type=directions.device.type, enabled=False):
        for start in range(0, directions.shape[0], chunk):
            stop = start + chunk
            d = directions32[start:stop]
            w = weights32[start:stop]
            cosine = torch.bmm(d, d.transpose(-1, -2)).clamp(-1.0, 1.0)
            kernel = torch.arccos(cosine).square_().add_(scale * scale).reciprocal_()
            pair = valid[start:stop, :, None] * valid[start:stop, None, :]
            # a particle is not its own pair: the j == i term is 1 / scale^2 for everyone
            # and would just reinstate the momentum weighting this weight exists to replace
            pair = pair * (1.0 - torch.eye(d.shape[1], dtype=pair.dtype, device=pair.device))
            out.append(w * torch.bmm(kernel * pair, w.unsqueeze(-1)).squeeze(-1))
    collinearity = torch.cat(out, dim=0).to(reference)
    return collinearity / collinearity.sum(dim=-1, keepdim=True).clamp_min(_AXIS_EPS)


def get_collinear_axis(
    fourmomenta: torch.Tensor, mask: torch.Tensor, scale: float = 0.02
) -> torch.Tensor:
    """Single event axis minimizing the half-precision damage; ``(B, 3)`` unit vectors.

    Same construction as :func:`get_principal_axis` -- leading eigenvector of
    ``sum_i c_i n_i n_i^T`` -- with :func:`collinearity_weights` in place of ``|p_i|``.
    """
    weights, directions = _weighted_directions(fourmomenta, mask)
    collinearity = collinearity_weights(directions, weights, weights > 0, scale)
    return leading_eigenvector(_direction_tensor(collinearity, directions))


class LGATrSlimTagger(nn.Module):
    """Weaver-facing L-GATr-slim jet tagger.

    Dense (zero-padded) port of the tagging-guide ``LGATrSlimWrapper`` with identity frames:
    the input four-momenta are embedded as Lorentz-vector channels, symmetry-breaking spurions
    and (optionally) a global class token are prepended, and the per-jet logits are read off
    the class token's scalar output (or a masked mean when ``mean_aggregation=True``).

    By default (``auxiliary_scalars=None``) the kinematic scalar features (log pt, log E,
    ...) are expected to be provided as part of the input features via the weaver data
    config (as for ParticleTransformer). Set ``auxiliary_scalars='all'`` to instead compute
    the seven standardized tagging features (log pt, log E, log pt_rel, log E_rel, dphi,
    deta, dr) internally from the four-momenta and prepend them to the input scalars; they
    must then NOT also be listed in ``pf_features``.

    Parameters
    ----------
    input_dim
        Number of input scalar features per particle.
    num_classes
        Number of output classes.
    hidden_v_channels / hidden_s_channels / num_blocks / num_heads / mlp_ratio / attn_ratio /
    num_layers_mlp / nonlinearity / nonlinearity_v / dropout_prob / norm_elementwise_affine /
    checkpoint_blocks / naive_amp / gate_norm / gate_scale_init
        Forwarded to :class:`LGATrSlim` (defaults follow the tagging-guide ``tag_slim`` config).
    compile_model / compile_kwargs
        Accepted for interface compatibility but no longer used to self-compile: weaver's
        ``--compile`` wraps the whole model, so compiling here too would compile twice.
        Pass torch.compile options through weaver's ``--compiler-option`` instead.
        ``compile_model`` still switches the trimmer to ``round_to_32``.
    beam_reference / two_beams / add_time_reference / spurion_scale
        Spurion configuration (defaults follow the tagging-guide ``tagging`` config).
    auxiliary_scalars
        Which kinematic features to compute from the four-momenta and prepend to the input
        scalars: 'all' (the seven standardized tagging features), 'zinvariant',
        'so3invariant', or None (default) to compute none of them, in which case the
        four-momenta are only used as Lorentz-vector inputs.
    momentum_float64
        Whether to compute the kinematic features in float64 (the tagging-guide default).
        Unused when ``auxiliary_scalars=None``. ONNX export requires ``False``
        (onnxruntime lacks float64 kernels for some of the involved ops, e.g. Atan).
    mean_aggregation
        If ``True``, aggregate with a masked mean over tokens instead of a class token.
    vector_units
        The four-momenta are divided by this scale (e.g. 20.0 to convert GeV to units of 20 GeV);
        spurions are not rescaled.
    attention_backend
        ``"auto"`` (default) chooses per forward call: on CUDA ``"efficient"`` when attention
        runs in fp32 (no AMP) and ``"varlen"`` under fp16/bf16 AMP (``"flex"`` if torch has no
        ``varlen_attn``); on CPU and during ONNX export ``"native"``. The two CUDA defaults are
        the fastest backend at each precision (JetClass / hh4b fp32 step: efficient 48.8 /
        176.6 ms vs flex 64.4 / 245.7 ms; fp16: varlen 24.9 / 68.0 ms vs efficient 29.2 /
        80.4 ms, RTX 5090, ``--compile``).
        ``"native"`` runs the dense zero-padded layout through
        ``torch.nn.functional.scaled_dot_product_attention``. ``"efficient"`` (torch's
        memory-efficient cutlass kernel with ``cu_seqlens``; exact fp32 path, graph break
        under compile), ``"varlen"`` (torch's native
        flash-attention varlen kernel, torch >= 2.10), ``"flash"`` (the flash-attn
        package, FlashAttention-3 interface preferred), ``"xformers"``
        (``xformers.ops.memory_efficient_attention`` with a block-diagonal mask), and
        ``"flex"`` (``torch.nn.attention.flex_attention``, torch >= 2.5) drop
        the padding and run block-diagonal attention over the packed tokens instead.
        These packed backends require CUDA; on CPU the packed layout falls back to a
        materialized block-diagonal SDPA mask. ONNX export requires ``"native"``.

        The first three run attention in fp16/bf16 (their kernels have no fp32 path). In
        Cartesian coordinates (``vector_coord="cartesian"``) that costs ~1 pp of accuracy against
        fp32 ``"native"``, while ``"flex"`` follows the trainer's precision and in fp32
        matches ``"native"`` numerically (to ~1e-6 on logits and gradients) -- faster and
        much lighter on memory. In the default light-cone frame half-precision attention
        keeps fp32 accuracy.

        One fwd+bwd step at batch 512 on an RTX 5090 (8 blocks, 128 particles), with
        ``--compile`` / without:

        =========  ==================  ==================
        backend    compiled            eager
        =========  ==================  ==================
        native      81 ms / 12.0 GiB   142 ms / 14.1 GiB
        flex        70 ms /  5.8 GiB    88 ms /  6.9 GiB
        varlen      39 ms /  5.2 GiB    66 ms /  6.3 GiB
        =========  ==================  ==================
    vector_coord
        ``"lightcone"`` (default) maps every vector, spurions included, to per-jet
        light-cone coordinates (:func:`get_lightcone_frame`): the same function in exact
        arithmetic, but well-conditioned, so attention and the vector GEMMs can run in the
        AMP dtype -- fp16 under ``--use-amp``, bf16 with ``--amp-dtype bf16``. With the
        flex backend that cuts the step time of AMP training ~2.3x (143 -> ~61 ms at batch
        2048 on an RTX 5090) at fp32 accuracy (2-epoch JetClass). ``"cartesian"`` feeds the
        four-momenta in Cartesian ``(E, px, py, pz)`` coordinates, where half precision
        destroys the Minkowski products, so train those without ``--use-amp``.
    lightcone_frame
        Which light-cone axis the coordinates are built on; one frame per event in every
        case, so attention is unchanged.

        ``"sum"`` (default) is the per-jet axis of :func:`get_lightcone_frame`: the summed
        constituent momentum. It assumes a jet, and on a full event it points along the
        beam and protects only the forward particles.

        ``"principal"`` uses the event principal axis instead -- the leading eigenvector
        of the weighted direction tensor ``sum_i |p_i| n_i n_i^T`` -- which is defined for
        any set of particles, including a balanced dijet whose momentum sum vanishes, and
        is built with the degeneracy-free :func:`lightcone_frame_from_axis`. See
        :func:`get_principal_axis`.

        ``"collinear"`` solves the same eigenproblem but weights each particle by how much
        *nearby* momentum it has (:func:`collinearity_weights`) instead of by its own
        momentum. That is the weight the conditioning objective actually asks for, since a
        pair's half-precision error scales as ``(theta_to_axis / theta_pair)^2``: a
        collimated cluster, where every internal pair is ill-conditioned, outvotes an
        isolated hard particle, whose pairs are all well conditioned anyway. On hh4b events
        it roughly halves the weighted product error of ``"principal"``, for one O(N^2)
        kernel per batch (+5% step time). See :func:`get_collinear_axis` and
        ``lightcone_collinear_scale``.

        On 10-epoch hh4b full-event trainings with fp16 AMP and the varlen backend, all
        three are within 0.03 pp of Cartesian fp32 at ~3.5x its speed, and ~0.15 pp ahead
        of Cartesian coordinates at the same precision.
    lightcone_collinear_scale
        Angular scale (rad) below which a pair counts as collinear in the ``"collinear"``
        weights; also the regulator as ``theta_ij -> 0``. Default 0.02 rad (~1 degree).
        On hh4b the weighted product error is flat over 0.002-0.02 and degrades above
        0.05, so the default sits at the loose end of the plateau, where it still
        regulates genuinely coincident directions.
    trim
        Whether to enable sequence trimming during training.
    use_amp
        Accepted for weaver compatibility (``train.py`` passes it whenever ``--use-amp``
        is given) and deliberately ignored: autocast is entered centrally, once per step,
        by ``weaver.utils.nn.tools``. A model-side ``torch.autocast(enabled=use_amp)``
        would not merely duplicate that -- with ``use_amp=False`` it *disables* autocast
        for the whole forward and overrides the trainer's region, so a model constructed
        outside weaver silently ran fp32 while claiming AMP.
    for_inference
        Whether to apply a softmax to the output (for deployment/ONNX export).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        # network configurations
        hidden_v_channels: int = 32,
        hidden_s_channels: int = 96,
        num_blocks: int = 12,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        attn_ratio: int = 1,
        num_layers_mlp: int = 2,
        nonlinearity: str = "gelu",
        nonlinearity_v: str | None = "sigmoid",
        dropout_prob: float | None = None,
        norm_elementwise_affine: bool = True,
        # spurions
        beam_reference: str | None = "all",
        two_beams: bool = True,
        add_time_reference: bool = True,
        spurion_scale: float = 1.0,
        # embedding, aggregation and scaling
        auxiliary_scalars: str | None = None,
        momentum_float64: bool = True,
        mean_aggregation: bool = False,
        vector_units: float = 1.0,
        # attention
        attention_backend: str = "auto",
        # vector coordinates (see get_lightcone_frame)
        vector_coord: str = "lightcone",
        lightcone_frame: str = "sum",
        lightcone_collinear_scale: float = 0.02,
        # misc
        gate_norm: str = "none",
        gate_scale_init: float = 4.0,
        checkpoint_blocks: bool = False,
        naive_amp: bool = False,
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
        trim: bool = True,
        use_amp: bool = False,
        for_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        _logger.info("LGATrSlimTagger init-ed: %s", locals())

        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention_backend: {attention_backend}. "
                f"Supported backends: {ATTENTION_BACKENDS}."
            )
        if vector_coord not in ("lightcone", "cartesian"):
            raise ValueError(
                f"Unsupported vector_coord: {vector_coord}. Supported: 'lightcone', 'cartesian'."
            )
        if lightcone_frame not in LIGHTCONE_FRAMES:
            raise ValueError(
                f"Unsupported lightcone_frame: {lightcone_frame}. Supported: {LIGHTCONE_FRAMES}."
            )
        if lightcone_frame != "sum" and vector_coord != "lightcone":
            raise ValueError(
                f"lightcone_frame={lightcone_frame!r} needs vector_coord='lightcone'; "
                "in Cartesian coordinates there is no frame to choose."
            )

        self.auxiliary_scalars = auxiliary_scalars
        self.momentum_float64 = momentum_float64
        self.mean_aggregation = mean_aggregation
        self.vector_units = vector_units
        self.attention_backend = attention_backend
        # accepted and ignored; weaver enters autocast centrally (see the class docstring)
        self.use_amp = use_amp
        self.for_inference = for_inference
        self.vector_coord = vector_coord
        self.lightcone_frame = lightcone_frame
        self.lightcone_collinear_scale = lightcone_collinear_scale

        spurions = get_spurion(beam_reference, add_time_reference, two_beams) * spurion_scale
        self.register_buffer("spurions", spurions, persistent=False)

        # the network sees the kinematic features, the extra scalars, and (unless
        # mean-aggregating) one extra channel flagging the global class token
        in_s_channels = (
            get_num_auxiliary_scalars(auxiliary_scalars)
            + input_dim
            + (0 if mean_aggregation else 1)
        )
        self.net = LGATrSlim(
            num_blocks=num_blocks,
            in_v_channels=1,
            out_v_channels=0,
            hidden_v_channels=hidden_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=num_classes,
            hidden_s_channels=hidden_s_channels,
            num_heads=num_heads,
            nonlinearity=nonlinearity,
            nonlinearity_v=nonlinearity_v,
            mlp_ratio=mlp_ratio,
            attn_ratio=attn_ratio,
            num_layers_mlp=num_layers_mlp,
            dropout_prob=dropout_prob,
            norm_elementwise_affine=norm_elementwise_affine,
            checkpoint_blocks=checkpoint_blocks,
            naive_amp=naive_amp,
            lightcone=vector_coord == "lightcone",
            gate_norm=gate_norm,
            gate_scale_init=gate_scale_init,
            # NOT compile=compile_model: weaver's --compile already wraps the whole
            # model (train.py), so self-compiling here would compile twice. The flag is
            # kept as a signal that the model is being compiled externally.
            compile=False,
        )
        # The flex backend sizes its block mask from the head dim and the attention dtype
        # (see _flex_block_size and _attention_dtypes). Head dim: q/k/v concatenate 4
        # components per hidden vector channel with the scalar channels, as in
        # SlimSelfAttention._pre_attention_reshape.
        attn = self.net.blocks[0].attention
        self._flex_head_dim = 4 * attn.hidden_v_channels + attn.hidden_s_channels
        if compile_kwargs:
            _logger.warning(
                "compile_kwargs=%s is ignored: the model is compiled as a whole by "
                "weaver's --compile; pass torch.compile options via --compiler-option.",
                compile_kwargs,
            )

        # Reserve the tokens prepended in forward() (spurions, plus the global class
        # token) so that under torch.compile the trimmer yields a sequence length that is
        # a clean multiple of 32 *after* they are added. Without this the compiled graph
        # sees a fresh derived symbol on every trimmed batch, which on torch 2.11 trips
        # dynamic-shape bugs in Inductor. Mirrors ParticleTransformer.
        num_extra_tokens = spurions.size(0) + (0 if mean_aggregation else 1)
        self.trimmer = SequenceTrimmer(
            enabled=trim and not for_inference,
            round_to_32=compile_model,
            num_extra_tokens=num_extra_tokens,
        )

    def _lightcone_frame(self, fourmomenta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The ``(N, 4, 4)`` float64 light-cone frame of each event, per ``lightcone_frame``.

        Pure geometry, so the caller computes it outside the autograd graph; the momentum
        transform it feeds stays differentiable.
        """
        if self.lightcone_frame == "sum":
            # per-jet axis: the summed constituent momentum
            return get_lightcone_frame(fourmomenta, mask)
        momenta = fourmomenta.to(torch.float64)
        if self.lightcone_frame == "collinear":
            axis = get_collinear_axis(momenta, mask, self.lightcone_collinear_scale)
        else:
            axis = get_principal_axis(momenta, mask)
        return lightcone_frame_from_axis(axis)

    def _check_amp(self, device_type: str) -> None:
        """Warn, once per process, about the two AMP settings known to cost accuracy.

        This has to look at the autocast state in the forward, since the precision is the
        trainer's and the model is never told the AMP dtype. Under ``torch.compile`` the
        logger call is a graph break on the first trace only: once the setting is in
        ``_AMP_WARNED`` the branch is skipped and the frame recompiles without it.
        """
        if not _autocast_active(device_type):
            return
        dtype = torch.get_autocast_dtype(device_type)
        if dtype not in (torch.float16, torch.bfloat16):
            return
        if self.vector_coord == "cartesian" and "cartesian" not in _AMP_WARNED:
            _AMP_WARNED.add("cartesian")
            _logger.warning(
                "LGATrSlimTagger runs under %s AMP with vector_coord='cartesian': half "
                "precision destroys the Minkowski products of Cartesian four-momenta. Use "
                "vector_coord='lightcone' (the default), or train without --use-amp.",
                dtype,
            )
        if dtype == torch.bfloat16 and "bf16" not in _AMP_WARNED:
            _AMP_WARNED.add("bf16")
            _logger.warning(
                "LGATrSlimTagger runs under bf16 AMP: bf16's 8-bit mantissa costs accuracy "
                "here, and fp16 is just as fast. Use --amp-dtype fp16."
            )

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P) -- scalar features
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        self._check_amp(x.device.type)
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)
        mask = mask.squeeze(1)  # (N, P) bool

        scalars = x.transpose(1, 2)  # (N, P, C)
        # (E, px, py, pz) convention, in units of `vector_units`
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        frame = None
        if self.vector_coord == "lightcone":
            with torch.no_grad():
                frame = self._lightcone_frame(fourmomenta, mask)  # (N, 4, 4) float64
            vectors = torch.einsum("nij,npj->npi", frame, fourmomenta.to(torch.float64))
        else:
            vectors = fourmomenta
        vectors = vectors.to(scalars.dtype) / self.vector_units
        # Zero out padded entries (data configs may pad by wrapping real particles).
        # ``where``, not a multiplication by the mask: padding is allowed to hold
        # anything, and 0 * nan is nan.
        keep = mask.unsqueeze(-1)
        vectors = torch.where(keep, vectors, torch.zeros_like(vectors))
        scalars = torch.where(keep, scalars, torch.zeros_like(scalars))

        if self.auxiliary_scalars is not None:
            # kinematic features, computed in the original (unscaled) momentum units
            if self.momentum_float64:
                fourmomenta = fourmomenta.to(torch.float64)
            fourmomenta = torch.where(
                mask.unsqueeze(-1), fourmomenta, torch.zeros_like(fourmomenta)
            )
            jet = fourmomenta.sum(dim=1, keepdim=True)  # (N, 1, 4)
            aux = get_auxiliary_scalars(
                fourmomenta, jet, auxiliary_scalars=self.auxiliary_scalars
            )
            aux = aux.to(scalars.dtype) * mask.unsqueeze(-1)
            scalars = torch.cat([aux, scalars], dim=-1)

        batch_size = x.size(0)
        # prepend spurions (zero scalar features, valid mask)
        if frame is not None:
            # the spurions are vectors like any other: same frame, not rescaled
            spurions = torch.einsum("nij,sj->nsi", frame, self.spurions.to(torch.float64))
            spurions = spurions.to(scalars.dtype)
        else:
            spurions = self.spurions.to(scalars.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        vectors = torch.cat([spurions, vectors], dim=1)
        scalars = torch.cat(
            [scalars.new_zeros(batch_size, spurions.size(1), scalars.size(2)), scalars], dim=1
        )
        mask = torch.cat(
            [mask.new_ones(batch_size, spurions.size(1)), mask], dim=1
        )

        backend = self._resolve_backend(x.device.type)
        if backend != "native":
            return self._forward_packed(vectors, scalars, mask, backend)

        if not self.mean_aggregation:
            # prepend a global class token: zero vector, one-hot flag in an extra scalar channel
            vectors = torch.cat([torch.zeros_like(vectors[:, :1]), vectors], dim=1)
            new_s = scalars.new_zeros(batch_size, scalars.size(1) + 1, scalars.size(2) + 1)
            new_s[:, 1:, :-1] = scalars
            new_s[:, 0, -1] = 1.0
            scalars = new_s
            mask = torch.cat([mask.new_ones(batch_size, 1), mask], dim=1)

        vectors = vectors.unsqueeze(2)  # (N, 1 + S + P, 1, 4)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, 1 + S + P)

        _, out = self.net(vectors, scalars, attn_mask=attn_mask)
        # out: (N, 1 + S + P, num_classes)

        if self.mean_aggregation:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
            output = out.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
        else:
            output = out[:, 0]

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output

    def forward_jagged(self, x, v, offsets, maxlen):
        """Forward from unpadded inputs, as delivered by the ``--jagged nested`` loader.

        x: (T, C) scalar features and v: (T, 4) [px,py,pz,energy] of all T particles of
        the batch, event after event; offsets: (N + 1,) event boundaries; ``maxlen``: any
        upper bound on the particles per event (a Python int, so that no device-to-host
        sync is needed). Computes what :meth:`forward` computes with a packed attention
        backend and ``trim=False``, without ever materializing the padded layout.
        """
        backend = self._resolve_backend(x.device.type)
        if backend == "native":
            if self.attention_backend != "auto":
                raise ValueError("forward_jagged needs a packed attention backend, not 'native'.")
            # "auto" on CPU: the packed layout falls back to a materialized block-diagonal mask
            backend = "efficient"
        if self.vector_coord == "lightcone" and self.lightcone_frame == "collinear":
            raise NotImplementedError("the collinear axis needs the dense (N, P, P) pair kernel.")
        self._check_amp(x.device.type)
        num_events = offsets.numel() - 1
        num_particles = x.shape[0]
        batch = get_batch_from_ptr(offsets, num_items=num_particles)  # (T,) event index

        scalars = x
        fourmomenta = v[:, [3, 0, 1, 2]]  # (E, px, py, pz)
        frame = None
        if self.vector_coord == "lightcone":
            with torch.no_grad():
                momenta = fourmomenta.to(torch.float64)
                if self.lightcone_frame == "sum":
                    p = momenta.new_zeros(num_events, 3).index_add_(0, batch, momenta[:, 1:])
                    frame = _lightcone_frame_from_sum(p)
                else:
                    spatial = momenta[:, 1:]
                    magnitude = spatial.norm(dim=-1)
                    directions = torch.where(
                        (magnitude > 0).unsqueeze(-1),
                        spatial / magnitude.clamp_min(_AXIS_EPS).unsqueeze(-1),
                        torch.zeros_like(spatial),
                    )
                    total = magnitude.new_zeros(num_events).index_add_(0, batch, magnitude)
                    weights = magnitude / total.clamp_min(_AXIS_EPS)[batch]
                    outer = weights[:, None, None] * directions[:, :, None] * directions[:, None, :]
                    tensor = outer.new_zeros(num_events, 3, 3).index_add_(0, batch, outer)
                    frame = lightcone_frame_from_axis(leading_eigenvector(tensor))  # (N, 4, 4)
            vectors = torch.einsum("tij,tj->ti", frame[batch], fourmomenta.to(torch.float64))
        else:
            vectors = fourmomenta
        vectors = vectors.to(scalars.dtype) / self.vector_units

        if self.auxiliary_scalars is not None:
            if self.momentum_float64:
                fourmomenta = fourmomenta.to(torch.float64)
            jet = fourmomenta.new_zeros(num_events, 4).index_add_(0, batch, fourmomenta)
            aux = get_auxiliary_scalars(fourmomenta, jet[batch], auxiliary_scalars=self.auxiliary_scalars)
            scalars = torch.cat([aux.to(scalars.dtype), scalars], dim=-1)

        # prepend the spurions of every event (zero scalar features), still packed
        num_spurions = self.spurions.size(0)
        if num_spurions > 0:
            if frame is not None:
                spurions = torch.einsum("nij,sj->nsi", frame, self.spurions.to(torch.float64))
                spurions = spurions.to(scalars.dtype)
            else:
                spurions = self.spurions.to(scalars.dtype).unsqueeze(0).expand(num_events, -1, -1)
            steps = torch.arange(num_events + 1, device=offsets.device) * num_spurions
            ptr = offsets + steps
            num_total = num_particles + num_events * num_spurions
            particle_idxs = torch.arange(num_particles, device=offsets.device) + (batch + 1) * num_spurions
            spurion_idxs = (
                ptr[:-1].unsqueeze(1) + torch.arange(num_spurions, device=offsets.device)
            ).flatten()
            vectors = (
                vectors.new_zeros(num_total, 4)
                .index_put((particle_idxs,), vectors)
                .index_put((spurion_idxs,), spurions.reshape(-1, 4))
            )
            scalars = scalars.new_zeros(num_total, scalars.shape[-1]).index_put((particle_idxs,), scalars)
            batch = get_batch_from_ptr(ptr, num_items=num_total)
        else:
            ptr = offsets
        return self._forward_tokens(vectors, scalars, batch, ptr, maxlen + num_spurions, backend)

    def _resolve_backend(self, device_type: str) -> str:
        """The attention backend for this call: ``attention_backend`` itself, or for
        ``"auto"`` the fastest backend at the precision attention will run in."""
        if self.attention_backend != "auto":
            return self.attention_backend
        if device_type != "cuda" or torch.onnx.is_in_onnx_export():
            return "native"
        if self._attention_dtypes(device_type)[0] == torch.float32:
            return "efficient"
        return "varlen" if _has_varlen_attn() else "flex"

    def _attention_dtypes(self, device_type: str) -> tuple[torch.dtype, ...]:
        """The precision attention will run in, for sizing the flex block mask (see
        :func:`_flex_block_size`): the trainer's autocast dtype under AMP, else fp32."""
        if _autocast_active(device_type):
            return (torch.get_autocast_dtype(device_type),)
        return (torch.float32,)

    def _forward_packed(self, vectors, scalars, mask, backend):
        """Packed (sparse) forward path: drop the padding and run block-diagonal varlen
        attention over the concatenated tokens (port of the tagging-guide
        ``LGATrWrapper._forward_sparse``)."""
        # any upper bound on the per-event sequence lengths works; using the dense width
        # avoids a device-to-host sync
        maxlen = mask.size(1)
        [vectors, scalars], batch, ptr = dense_to_sparse([vectors, scalars], mask)
        return self._forward_tokens(vectors, scalars, batch, ptr, maxlen, backend)

    def _forward_tokens(self, vectors, scalars, batch, ptr, maxlen, backend):
        """Packed ``(tokens, 4)`` vectors / ``(tokens, C)`` scalars (spurions included)
        -> logits. ``maxlen`` is any upper bound on the per-event token count."""
        if not self.mean_aggregation:
            # prepend a global class token per event: zero vector, one-hot flag in an
            # extra scalar channel
            maxlen = maxlen + 1
            global_idxs, nonglobal_idxs, ptr, batch, num_total = insert_global_tokens(
                ptr, batch, vectors.shape[0]
            )
            vectors = vectors.new_zeros(num_total, vectors.shape[-1]).index_put(
                (nonglobal_idxs,), vectors
            )
            # Built column-wise and concatenated rather than by writing into a wider
            # zero tensor: mutating a view in place (``new_s[:, -1].index_fill_(...)``)
            # functionalizes to a ``copy_`` into an in-graph ``empty``, which Inductor
            # asserts on, so it broke end-to-end torch.compile of the packed path.
            flag = scalars.new_zeros(num_total, 1).index_fill(0, global_idxs, 1.0)
            scalars = torch.cat(
                [
                    scalars.new_zeros(num_total, scalars.shape[-1]).index_put(
                        (nonglobal_idxs,), scalars
                    ),
                    flag,
                ],
                dim=-1,
            )

        vectors = vectors.unsqueeze(0).unsqueeze(-2)  # (1, tokens, 1, 4)
        scalars = scalars.unsqueeze(0)  # (1, tokens, C)
        device_type = vectors.device.type
        # inside the trainer's autocast region, where the attention dtype is known
        attn_kwargs = get_sparse_attention_kwargs(
            ptr,
            batch,
            maxlen,
            backend,
            flex_head_dim=self._flex_head_dim,
            flex_dtypes=self._attention_dtypes(device_type),
        )
        _, out = self.net(vectors, scalars, **attn_kwargs)
        out = out.squeeze(0)  # (tokens, num_classes)

        if self.mean_aggregation:
            batch_size = ptr.numel() - 1
            counts = (ptr[1:] - ptr[:-1]).unsqueeze(-1).to(out.dtype)
            output = out.new_zeros(batch_size, out.shape[-1]).index_add_(0, batch, out) / counts
        else:
            output = out.index_select(0, global_idxs)

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output
