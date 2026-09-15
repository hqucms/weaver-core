"""Point Transformer V3 -- pure PyTorch port for weaver.

Ported from the detached PTv3 release (https://github.com/Pointcept/PointTransformerV3,
``model.py`` + ``serialization/``; Xiaoyang Wu et al., "Point Transformer V3: Simpler,
Faster, Stronger", CVPR 2024, arXiv:2312.10035). Please cite the original work.

The original depends on spconv, torch_scatter, timm, addict and (optionally) flash-attn.
This port keeps the architecture, the module/parameter structure and the defaults of
``PointTransformerV3`` but only needs PyTorch:

* ``spconv.SubMConv3d`` (used by the embedding stem and the conditional positional
  encoding, CPE) is replaced by :class:`SubMConv3d`, a submanifold sparse convolution
  on a hashed voxel grid (sorted voxel codes + ``searchsorted`` neighbour lookup). For
  point clouds with one point per voxel it is exactly spconv's submanifold conv. Point
  clouds with several points per voxel (common for neutrino telescopes: all pulses of an
  optical module share its position) are handled explicitly -- the centre tap acts on the
  point's own features and the off-centre taps on the voxel-averaged features of the
  neighbouring voxels -- whereas spconv silently picks one point per duplicated site.
* ``torch_scatter.segment_csr`` is replaced by ``Tensor.scatter_reduce``.
* ``timm``'s ``DropPath`` and ``addict.Dict`` are re-implemented inline.
* Serialized attention always uses the patch grouping of the flash-attention path of
  the original (point clouds larger than ``patch_size`` are padded to a multiple of it by
  re-using points of the previous patch; smaller ones form a single shorter patch). The
  attention itself runs with ``attn_backend``:

    - ``"native"`` (default): ``F.scaled_dot_product_attention`` on patches padded to the
      longest patch in the batch with a key-padding mask; fp32 capable, supports RPE and
      ``upcast_attention`` / ``upcast_softmax`` (then evaluated explicitly);
    - ``"varlen"``: torch's native flash-attention varlen kernel (torch >= 2.10, CUDA);
    - ``"flash"``: ``flash_attn.flash_attn_varlen_qkvpacked_func`` (as the original with
      ``enable_flash=True``, which is accepted as an alias).

  The original non-flash fallback, which shrinks the patch size of the whole batch to the
  smallest point cloud in it, is not reproduced: it would make every patch as small as
  the smallest event.
* Determinism: the serialization orders are only shuffled in training mode, and the
  serialization depth can be fixed (``serialization_depth``), so that evaluation outputs
  of a point cloud do not depend on the rest of the batch.

:class:`PointTransformerV3` keeps the Pointcept interface (a dict with ``feat``,
``grid_coord`` or ``coord`` + ``grid_size``, and ``offset`` or ``batch``).
:class:`PTv3EventRegressor` wraps it for weaver: it takes padded ``(N, C, P)`` point
inputs plus a mask, runs the PTv3 encoder, pools each point cloud, and returns an
``(N, num_outputs)`` tensor for event-level regression/classification.
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ---------------------------------------------------------------------------------------
# Serialization (z-order and Hilbert curves), from PTv3's `serialization/`
# ---------------------------------------------------------------------------------------


class _KeyLUT:
    """Look-up tables for z-order (Morton) encoding. From OCNN (Peng-Shuai Wang, MIT)."""

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        cpu = torch.device("cpu")
        self._encode = {
            cpu: (
                self._xyz2key(r256, zero, zero, 8),
                self._xyz2key(zero, r256, zero, 8),
                self._xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {cpu: self._key2xyz(r512, 9)}

    def encode_lut(self, device):
        if device not in self._encode:
            self._encode[device] = tuple(e.to(device) for e in self._encode[torch.device("cpu")])
        return self._encode[device]

    def decode_lut(self, device):
        if device not in self._decode:
            self._decode[device] = tuple(e.to(device) for e in self._decode[torch.device("cpu")])
        return self._decode[device]

    @staticmethod
    def _xyz2key(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = key | ((x & mask) << (2 * i + 2)) | ((y & mask) << (2 * i + 1)) | ((z & mask) << (2 * i + 0))
        return key

    @staticmethod
    def _key2xyz(key, depth):
        x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = _KeyLUT()


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    EX, EY, EZ = _key_lut.encode_lut(grid_coord.device)
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


def z_order_decode(code: torch.Tensor, depth: int = 16) -> torch.Tensor:
    DX, DY, DZ = _key_lut.decode_lut(code.device)
    x, y, z = torch.zeros_like(code), torch.zeros_like(code), torch.zeros_like(code)
    code = code & ((1 << 48) - 1)
    for i in range((depth + 2) // 3):
        k = code >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))
    return torch.stack([x, y, z], dim=-1)


def _right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return F.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)


def _binary2gray(binary, axis=-1):
    return torch.logical_xor(binary, _right_shift(binary, axis=axis))


def _gray2binary(gray, axis=-1):
    shift = 2 ** (int(math.ceil(math.log2(gray.shape[axis]))) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift //= 2
    return gray


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """Hilbert code of 3D grid coordinates (Skilling's algorithm, vectorized over points).

    Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve (via PTv3).
    """
    num_dims, num_bits = 3, depth
    assert grid_coord.shape[-1] == num_dims and num_dims * num_bits <= 63
    bitpack_mask = 1 << torch.arange(0, 8, device=grid_coord.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    locs_uint8 = grid_coord.long().contiguous().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = locs_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[..., -num_bits:]
    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    padded = F.pad(hh_bin, (64 - num_bits * num_dims, 0), "constant", 0)
    # (the original `.squeeze()` here collapsed a single-point input to a 0-d tensor)
    hh_uint8 = (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).type(torch.uint8)
    return hh_uint8.view(torch.int64).squeeze(-1)


def hilbert_decode(code: torch.Tensor, depth: int = 16) -> torch.Tensor:
    num_dims, num_bits = 3, depth
    code = torch.atleast_1d(code)
    bitpack_mask = 2 ** torch.arange(0, 8, device=code.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    hh_uint8 = code.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    hh_bits = (
        hh_uint8.unsqueeze(-1).bitwise_and(bitpack_mask_rev).ne(0).byte().flatten(-2, -1)[:, -num_dims * num_bits :]
    )
    gray = _binary2gray(hh_bits)
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)
    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)
    padded = F.pad(gray, (64 - num_bits, 0), "constant", 0)
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).type(torch.uint8)
    return locs_uint8.view(torch.int64).reshape(-1, num_dims)


@torch.no_grad()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    else:
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    if batch is not None:
        code = batch.long() << depth * 3 | code
    return code


# ---------------------------------------------------------------------------------------
# Batch bookkeeping and the Point structure
# ---------------------------------------------------------------------------------------


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch, num_batches: Optional[int] = None):
    return torch.cumsum(batch.bincount(minlength=num_batches or 0), dim=0).long()


def _ceil_div(a, b):
    return torch.div(a + b - 1, b, rounding_mode="floor")


def _segment_reduce(x, index, num_segments, reduce):
    """``torch_scatter.segment_csr`` / ``scatter`` replacement (segments given by ``index``)."""
    reduce = {"max": "amax", "min": "amin", "sum": "sum", "mean": "mean"}[reduce]
    out = x.new_zeros((num_segments,) + x.shape[1:])
    return out.scatter_reduce(0, index.view(-1, *([1] * (x.dim() - 1))).expand_as(x), x, reduce, include_self=False)


class Point(dict):
    """Point structure of Pointcept: a dict with attribute access (replaces ``addict.Dict``).

    Keys with a specific meaning:

    - ``coord``: original coordinates; ``grid_coord``: integer grid coordinates (or give
      ``coord`` + ``grid_size``);
    - ``offset`` / ``batch``: batch layout (points must be sorted by batch); one is derived
      from the other if missing;
    - ``feat``: point features;
    - ``serialized_depth`` / ``serialized_code`` / ``serialized_order`` /
      ``serialized_inverse``: serialization, see :meth:`serialization`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def _ensure_grid_coord(self):
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """Point cloud serialization. Relies on ``grid_coord`` (or ``coord`` + ``grid_size``) and ``batch``."""
        assert "batch" in self.keys()
        self._ensure_grid_coord()
        if depth is None:
            # adaptively measure the depth of the serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # maximum bit length for the serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = torch.stack([encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order])
        # stable: points sharing a grid cell keep their input order (e.g. time-ordered pulses)
        order = torch.argsort(code, stable=True)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order, src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1)
        )
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse


class PointModule(nn.Module):
    """Placeholder: all subclasses take a :class:`Point` in :class:`PointSequential`."""


class PointSequential(PointModule):
    """Sequential container; plain ``nn.Module``s are applied to ``point.feat``."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif isinstance(input, Point):
                input.feat = module(input.feat)
            else:
                input = module(input)
        return input


class DropPath(nn.Module):
    """Stochastic depth per sample (timm's ``DropPath``); here "sample" = row of ``feat``."""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


# ---------------------------------------------------------------------------------------
# Submanifold sparse convolution (replaces spconv.SubMConv3d)
# ---------------------------------------------------------------------------------------

_VOXEL_BITS = 17  # bits per axis in the packed voxel code (grid coords < 2**16, plus kernel reach)
_VOXEL_SHIFT = 4  # offset added to grid coords so that neighbours of cell 0 stay non-negative


@torch.no_grad()
def _submconv_indices(point: Point, kernel_size: int, indice_key: Optional[str]):
    """Voxelization and neighbour table of the active sites, cached per ``indice_key``.

    Returns ``(inverse, neighbours)``: ``inverse`` (N,) maps each point to its voxel,
    ``neighbours`` (V, kernel_size**3) holds the voxel index at each kernel offset (-1 if
    the neighbouring cell is empty). Voxels are keyed by (batch, grid_coord).
    """
    cache_key = f"_submconv_{indice_key}_{kernel_size}" if indice_key is not None else None
    if cache_key is not None and cache_key in point.keys():
        return point[cache_key]

    reach = kernel_size // 2
    grid = point.grid_coord.long() + _VOXEL_SHIFT
    assert int(grid.max()) + reach < (1 << _VOXEL_BITS), "grid_coord too large for the voxel hash"
    assert len(point.offset) < (1 << (63 - 3 * _VOXEL_BITS)), "too many point clouds in the batch"
    shift_x, shift_y = 2 * _VOXEL_BITS, _VOXEL_BITS
    code = (point.batch.long() << (3 * _VOXEL_BITS)) | (grid[:, 0] << shift_x) | (grid[:, 1] << shift_y) | grid[:, 2]
    voxels, inverse = torch.unique(code, sorted=True, return_inverse=True)

    r = torch.arange(-reach, reach + 1, device=code.device)
    dx, dy, dz = torch.meshgrid(r, r, r, indexing="ij")
    offsets = ((dx << shift_x) + (dy << shift_y) + dz).reshape(-1)  # packing is linear in each field
    query = voxels[:, None] + offsets[None, :]
    pos = torch.searchsorted(voxels, query).clamp_(max=len(voxels) - 1)
    neighbours = torch.where(voxels[pos] == query, pos, torch.full_like(pos, -1))

    result = (inverse, neighbours)
    if cache_key is not None:
        point[cache_key] = result
    return result


class SubMConv3d(PointModule):
    """Submanifold sparse 3D convolution on ``point.grid_coord`` (pure PyTorch).

    Outputs are computed at the input points only. The weight of the centre tap is
    applied to each point's own features; the other taps see the (mean) features of the
    points in the neighbouring voxels. With at most one point per voxel this equals
    ``spconv.SubMConv3d``. ``padding`` is accepted for API compatibility and ignored
    (it has no effect on a submanifold convolution).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, indice_key=None, padding=None):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.kernel_volume = kernel_size**3
        self.weight = nn.Parameter(torch.empty(self.kernel_volume, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # same scheme as torch.nn.Conv3d (kaiming_uniform with a=sqrt(5) => U(-1/sqrt(fan_in), ..))
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_volume)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, point: Point):
        inverse, neighbours = _submconv_indices(point, self.kernel_size, self.indice_key)
        feat = point.feat
        num_voxels = neighbours.shape[0]
        center = self.kernel_volume // 2

        out = feat @ self.weight[center]
        if self.kernel_volume > 1:
            if num_voxels == feat.shape[0]:
                # one point per voxel: `inverse` is a permutation
                voxel_feat = feat.new_empty(feat.shape).index_copy_(0, inverse, feat)
            else:
                voxel_feat = _segment_reduce(feat, inverse, num_voxels, "mean")
            nbr = torch.cat([neighbours[:, :center], neighbours[:, center + 1 :]], dim=1)
            gathered = voxel_feat[nbr.clamp(min=0)] * (nbr >= 0).unsqueeze(-1).to(feat.dtype)
            weight = torch.cat([self.weight[:center], self.weight[center + 1 :]], dim=0)
            voxel_out = gathered.reshape(num_voxels, -1) @ weight.reshape(-1, self.out_channels)
            out = out + voxel_out[inverse]
        if self.bias is not None:
            out = out + self.bias
        point.feat = out
        return point

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"bias={self.bias is not None}, indice_key={self.indice_key}"
        )


# ---------------------------------------------------------------------------------------
# PTv3 modules
# ---------------------------------------------------------------------------------------


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True))

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        condition = point.condition if isinstance(point.condition, str) else point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        return out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)


_ATTN_BACKENDS = ("native", "varlen", "flash")


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        attn_backend="native",
    ):
        super().__init__()
        assert channels % num_heads == 0
        if enable_flash:
            attn_backend = "flash"
        assert attn_backend in _ATTN_BACKENDS, attn_backend
        if attn_backend != "native":
            assert enable_rpe is False, "Set enable_rpe to False when using a varlen/flash attention kernel"
            assert upcast_attention is False, "Set upcast_attention to False when using a varlen/flash kernel"
            assert upcast_softmax is False, "Set upcast_softmax to False when using a varlen/flash kernel"
        if attn_backend == "flash":
            assert flash_attn is not None, "Make sure flash_attn is installed."
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.attn_backend = attn_backend
        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """Patch layout of the flash-attention path of PTv3, vectorized over the batch.

        Returns ``pad`` (padded position -> point), ``unpad`` (point -> padded position)
        and ``cu_seqlens`` (patch boundaries in the padded sequence, int32).
        """
        K = self.patch_size
        key = f"_patch_layout_{K}"
        if key not in point.keys():
            offset = point.offset
            device = offset.device
            bincount = offset2bincount(offset)
            # only pad point clouds with more points than patch_size
            bincount_pad = torch.where(bincount > K, _ceil_div(bincount, K) * K, bincount)
            _offset = F.pad(offset, (1, 0))
            _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            num_batches = len(bincount)
            ar = torch.arange(num_batches, device=device)

            batch_pad = ar.repeat_interleave(bincount_pad)
            local = torch.arange(len(batch_pad), device=device) - _offset_pad[batch_pad]
            # the padding slots at the end of a point cloud re-use the points one patch earlier
            pad = _offset[batch_pad] + local - K * (local >= bincount[batch_pad]).long()
            unpad = torch.arange(len(point.batch), device=device) + (_offset_pad - _offset)[point.batch]

            num_patches = _ceil_div(bincount_pad, K)
            patch_batch = ar.repeat_interleave(num_patches)
            patch_local = torch.arange(len(patch_batch), device=device) - F.pad(torch.cumsum(num_patches, 0), (1, 0))[
                patch_batch
            ]
            cu_seqlens = torch.cat([_offset_pad[patch_batch] + patch_local * K, _offset_pad[-1:]]).int()
            point[key] = (pad, unpad, cu_seqlens)
        return point[key]

    def _native_attention(self, qkv, cu_seqlens, grid_coord):
        H, C = self.num_heads, self.channels
        starts, lengths = cu_seqlens[:-1].long(), torch.diff(cu_seqlens).long()
        max_len = int(lengths.max())
        ar = torch.arange(max_len, device=qkv.device)
        valid = ar[None, :] < lengths[:, None]  # (P, L)
        index = (starts[:, None] + ar[None, :]).clamp_(max=qkv.shape[0] - 1)
        # (P, L, 3, H, C') => (3, P, H, L, C')
        q, k, v = qkv[index].reshape(len(starts), max_len, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        key_mask = valid[:, None, None, :]
        dropout_p = self.attn_drop if self.training else 0.0

        if self.enable_rpe or self.upcast_attention or self.upcast_softmax:
            if self.upcast_attention:
                q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (P, H, L, L)
            if self.enable_rpe:
                coord = grid_coord[index]
                attn = attn + self.rpe(coord.unsqueeze(2) - coord.unsqueeze(1))
            if self.upcast_softmax:
                attn = attn.float()
            attn = attn.masked_fill(~key_mask, float("-inf")).softmax(dim=-1)
            attn = F.dropout(attn, p=dropout_p, training=self.training).to(v.dtype)
            out = attn @ v
        else:
            mask = None if bool(valid.all()) else key_mask
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.scale)
        return out.transpose(1, 2).reshape(-1, max_len, C)[valid]

    def _varlen_attention(self, qkv, cu_seqlens):
        H, C = self.num_heads, self.channels
        max_len = int(torch.diff(cu_seqlens).max())
        if self.attn_backend == "flash":
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=max_len,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            )
            return out.reshape(-1, C).to(qkv.dtype)
        # torch >= 2.10 native varlen kernel, via the helper shared with the LGATr-slim port
        from weaver.nn.model.LGATrSlim import _varlen_attention

        assert self.attn_drop == 0 or not self.training, "attn_drop is not supported by the varlen backend"
        q, k, v = (t.reshape(-1, H, C // H).transpose(0, 1).unsqueeze(0) for t in qkv.chunk(3, dim=-1))
        if self.scale != (C // H) ** -0.5:
            q = q * (self.scale * (C // H) ** 0.5)
        out = _varlen_attention(q, k, v, cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens, max_q=max_len, max_k=max_len)
        return out.squeeze(0).transpose(0, 1).reshape(-1, C)

    def forward(self, point):
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat for serialized point patches
        qkv = self.qkv(point.feat)[order]
        if self.attn_backend == "native":
            grid_coord = point.grid_coord[order] if self.enable_rpe else None
            feat = self._native_attention(qkv, cu_seqlens, grid_coord)
        else:
            feat = self._varlen_attention(qkv, cu_seqlens)
        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
        attn_backend="native",
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            attn_backend=attn_backend,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {"serialized_code", "serialized_order", "serialized_inverse", "serialized_depth"}.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        # indices of points sorted by cluster
        _, indices = torch.sort(cluster, stable=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head indices of each cluster, for reducing attributes such as code, batch
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code, stable=True)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order, src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1)
        )
        if self.shuffle_orders and self.training:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]

        num_clusters = len(code_)
        batch = point.batch[head_indices]
        point_dict = dict(
            feat=_segment_reduce(self.proj(point.feat), cluster, num_clusters, self.reduce),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=batch,
            # keep empty point clouds (if any) so that the batch layout is preserved
            offset=batch2offset(batch, num_batches=len(point.offset)),
        )
        if "coord" in point.keys():
            point_dict["coord"] = _segment_reduce(point.coord, cluster, num_clusters, "mean")
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        return point


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(
            conv=SubMConv3d(in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem")
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)


class PointTransformerV3(PointModule):
    """Point Transformer V3 backbone (Pointcept interface, see module docstring).

    Differences in the constructor w.r.t. the original: ``enable_flash`` defaults to
    ``False``; new ``attn_backend`` (``native`` | ``varlen`` | ``flash``) and
    ``serialization_depth`` (``None`` = adaptive, as in the original).
    """

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        attn_backend="native",
        serialization_depth=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.serialization_depth = serialization_depth

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        act_layer = nn.GELU
        block_kwargs = dict(
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=ln_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            attn_backend=attn_backend,
        )

        self.embedding = Embedding(
            in_channels=in_channels, embed_channels=enc_channels[0], norm_layer=bn_layer, act_layer=act_layer
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        drop_path=enc_drop_path_[i],
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        **block_kwargs,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            drop_path=dec_drop_path_[i],
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            **block_kwargs,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """``data_dict`` must contain ``feat``, ``grid_coord`` (or ``coord`` + ``grid_size``) and ``offset`` or ``batch``."""
        point = Point(data_dict)
        point.serialization(
            order=self.order,
            depth=self.serialization_depth,
            shuffle_orders=self.shuffle_orders and self.training,
        )
        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point


# ---------------------------------------------------------------------------------------
# Event-level wrapper for weaver
# ---------------------------------------------------------------------------------------


class PTv3EventRegressor(nn.Module):
    """PTv3 encoder + per-event pooling + MLP head, on weaver's padded inputs.

    ``forward(points, features, mask, global_features=None)``:

    * ``points`` (N, 3, P): point coordinates (e.g. metres), used for voxelization and
      serialization: ``grid_coord = floor((points - min over the point cloud) / grid_size)``;
    * ``features`` (N, C, P): point features fed to the embedding stem;
    * ``mask`` (N, 1, P): 1 for real points, 0 for padding;
    * ``global_features`` (N, G) or (N, G, 1), optional: event-level features concatenated
      to the pooled representation.

    Returns (N, num_outputs). The backbone runs in ``cls_mode`` (encoder only) by default;
    with ``cls_mode=False`` the decoder output (at the input points) is pooled instead.
    """

    def __init__(
        self,
        input_dim,
        num_outputs,
        global_input_dim=0,
        grid_size=1.0,
        serialization_depth=16,
        pooling="mean",
        head_channels=(256, 128),
        head_dropout=0.0,
        cls_mode=True,
        **ptv3_kwargs,
    ):
        super().__init__()
        assert pooling in ("mean", "max", "mean+max")
        self.grid_size = grid_size
        self.pooling = pooling
        self.backbone = PointTransformerV3(
            in_channels=input_dim, cls_mode=cls_mode, serialization_depth=serialization_depth, **ptv3_kwargs
        )
        enc_channels = ptv3_kwargs.get("enc_channels", (32, 64, 128, 256, 512))
        dec_channels = ptv3_kwargs.get("dec_channels", (64, 64, 128, 256))
        feat_dim = enc_channels[-1] if cls_mode else dec_channels[0]
        in_dim = feat_dim * (2 if pooling == "mean+max" else 1) + global_input_dim
        layers = []
        for c in head_channels:
            layers += [nn.Linear(in_dim, c), nn.BatchNorm1d(c), nn.ReLU(inplace=True), nn.Dropout(head_dropout)]
            in_dim = c
        layers.append(nn.Linear(in_dim, num_outputs))
        self.head = nn.Sequential(*layers)

    def pack(self, points, features, mask):
        """(N, C, P) padded inputs -> Pointcept data dict (points sorted by batch)."""
        mask = mask.squeeze(1).bool()
        counts = mask.sum(1)
        batch = torch.arange(mask.shape[0], device=mask.device).repeat_interleave(counts)
        coord = points.transpose(1, 2)[mask].float()
        feat = features.transpose(1, 2)[mask]
        # per point cloud origin, so that the voxelization of an event does not depend on the batch
        coord_min = _segment_reduce(coord, batch, mask.shape[0], "min")
        grid_coord = torch.div(coord - coord_min[batch], self.grid_size, rounding_mode="floor").int()
        return dict(
            coord=coord, grid_coord=grid_coord, feat=feat, batch=batch, offset=torch.cumsum(counts, 0).long()
        )

    def forward(self, points, features, mask, global_features=None):
        num_events = mask.shape[0]
        point = self.backbone(self.pack(points, features, mask))
        pooled = []
        if "mean" in self.pooling:
            pooled.append(_segment_reduce(point.feat, point.batch, num_events, "mean"))
        if "max" in self.pooling:
            pooled.append(_segment_reduce(point.feat, point.batch, num_events, "max"))
        if global_features is not None:
            pooled.append(global_features.reshape(num_events, -1).to(pooled[0].dtype))
        return self.head(torch.cat(pooled, dim=1))
