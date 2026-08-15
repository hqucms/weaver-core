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
none of the LLoCa frames machinery is required. The attention backend is fixed to the
native ``torch.nn.functional.scaled_dot_product_attention`` (dense zero-padded path).
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

from weaver.nn.model.ParticleTransformer import SequenceTrimmer
from weaver.utils.logger import _logger


# ------------------------------------------------------------------------------------
# Autocast helpers (ported from lgatr/utils/autocast.py)
# ------------------------------------------------------------------------------------

# Toggled by the naive_amp context manager; read at call time so torch.compile constant-folds it.
_NAIVE_AMP = False


try:
    torch.is_autocast_enabled("cpu")

    def _autocast_active() -> bool:
        """Whether CPU or CUDA autocast is enabled."""
        return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")

except TypeError:  # pragma: no cover - torch<2.4 has no device_type argument

    def _autocast_active() -> bool:
        """Whether CPU or CUDA autocast is enabled."""
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


def _post_attention_reshape(
    out: torch.Tensor, hidden_v_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h_v = out[..., : hidden_v_channels * 4].unflatten(-1, (4, hidden_v_channels))
    h_s = out[..., hidden_v_channels * 4 :]

    h_v = _movedim(h_v, -4, -2).flatten(-2, -1)
    h_s = _movedim(h_s, -2, -3).flatten(-2, -1)
    return h_v, h_s


@minimum_autocast_precision(torch.float32, output="high")
def _call_attention(*args, **kwargs):
    return F.scaled_dot_product_attention(*args, **kwargs)


def _freeze_dead_tail(
    norm: nn.Module, mlp: nn.Module, out_v_channels: int, out_s_channels: int
) -> None:
    """Freeze last-block params that cannot receive grads when an output stream is empty."""
    if out_v_channels == 0:
        if norm.weight_v is not None:
            norm.weight_v.requires_grad_(False)
        for name, p in mlp.named_parameters():
            if name.endswith("weight_v"):
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

    @minimum_autocast_precision(torch.float32, output="high")
    def _linear_v(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.linear(vectors, self.weight_v)

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., 4, in_v_channels) -> (..., 4, out_v_channels)
        # scalars: (..., in_s_channels) -> (..., out_s_channels)
        outputs_v = self._linear_v(vectors)
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
        # 0.5 = 1/sqrt(4) controls the scale, like 1/sqrt(d_k) in attention
        return 0.5 * ((v_gates_1 * v_gates_2) * self.metric[..., None]).sum(dim=-2, keepdim=True)


class SlimSelfAttention(nn.Module):
    """Self-attention for Lorentz vectors and scalar features."""

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

        q_v = q_v * self.metric.to(q_v.dtype)[..., None]

        q = torch.cat([q_v.flatten(start_dim=-2), q_s], dim=-1)
        k = torch.cat([k_v.flatten(start_dim=-2), k_s], dim=-1)
        v = torch.cat([v_v.flatten(start_dim=-2), v_s], dim=-1)
        return q, k, v

    def forward(
        self, vectors: torch.Tensor, scalars: torch.Tensor, **attn_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., items, 4, v_channels); scalars: (..., items, s_channels)
        qkv_v, qkv_s = self.linear_in(vectors, scalars)

        q, k, v = self._pre_attention_reshape(qkv_v, qkv_s)
        out = _call_attention(q, k, v, **attn_kwargs)
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
        Whether to bypass the fp32 precision islands so the whole forward runs in the surrounding
        autocast dtype (e.g. bf16). When ``False`` (default), under autocast the vector stream and
        metric contractions stay fp32 while the scalar GEMMs run in bf16.
    compile
        Whether to wrap the model's forward with :func:`torch.compile`.
    compile_kwargs
        Dict forwarded verbatim to :func:`torch.compile` when ``compile=True``.
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
        compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self._naive_amp = naive_amp

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


class LGATrSlimTagger(nn.Module):
    """Weaver-facing L-GATr-slim jet tagger.

    Dense (zero-padded) port of the tagging-guide ``LGATrSlimWrapper`` with identity frames:
    the input four-momenta are embedded as Lorentz-vector channels, symmetry-breaking spurions
    and (optionally) a global class token are prepended, and the per-jet logits are read off
    the class token's scalar output (or a masked mean when ``mean_aggregation=True``).

    Kinematic scalar features (log pt, log E, ...) are expected to be provided as part of the
    input features via the weaver data config (as for ParticleTransformer), not computed here.

    Parameters
    ----------
    input_dim
        Number of input scalar features per particle.
    num_classes
        Number of output classes.
    hidden_v_channels / hidden_s_channels / num_blocks / num_heads / mlp_ratio / attn_ratio /
    num_layers_mlp / nonlinearity / nonlinearity_v / dropout_prob / norm_elementwise_affine /
    checkpoint_blocks / naive_amp / compile_model / compile_kwargs
        Forwarded to :class:`LGATrSlim` (defaults follow the tagging-guide ``tag_slim`` config).
    beam_reference / two_beams / add_time_reference / spurion_scale
        Spurion configuration (defaults follow the tagging-guide ``tagging`` config).
    mean_aggregation
        If ``True``, aggregate with a masked mean over tokens instead of a class token.
    vector_units
        The four-momenta are divided by this scale (e.g. 20.0 to convert GeV to units of 20 GeV);
        spurions are not rescaled.
    trim
        Whether to enable sequence trimming during training.
    use_amp
        Whether to run the network under ``torch.autocast``.
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
        # aggregation and scaling
        mean_aggregation: bool = False,
        vector_units: float = 1.0,
        # misc
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

        self.mean_aggregation = mean_aggregation
        self.vector_units = vector_units
        self.use_amp = use_amp
        self.for_inference = for_inference

        spurions = get_spurion(beam_reference, add_time_reference, two_beams) * spurion_scale
        self.register_buffer("spurions", spurions, persistent=False)

        # one extra scalar channel flags the global class token
        in_s_channels = input_dim + (0 if mean_aggregation else 1)
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
            compile=compile_model,
            compile_kwargs=compile_kwargs,
        )

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P) -- scalar features
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)
        mask = mask.squeeze(1)  # (N, P) bool

        scalars = x.transpose(1, 2)  # (N, P, C)
        # (E, px, py, pz) convention, in units of `vector_units`
        vectors = v.transpose(1, 2)[..., [3, 0, 1, 2]].to(scalars.dtype) / self.vector_units
        # zero out padded entries (data configs may pad by wrapping real particles)
        vectors = vectors * mask.unsqueeze(-1)
        scalars = scalars * mask.unsqueeze(-1)

        batch_size = x.size(0)
        # prepend spurions (zero scalar features, valid mask)
        spurions = self.spurions.to(scalars.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        vectors = torch.cat([spurions, vectors], dim=1)
        scalars = torch.cat(
            [scalars.new_zeros(batch_size, spurions.size(1), scalars.size(2)), scalars], dim=1
        )
        mask = torch.cat(
            [mask.new_ones(batch_size, spurions.size(1)), mask], dim=1
        )

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

        with torch.autocast(vectors.device.type, enabled=self.use_amp):
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
