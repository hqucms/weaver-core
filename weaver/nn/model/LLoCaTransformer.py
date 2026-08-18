"""LLoCa-Transformer: a Lorentz-equivariant transformer via local canonicalization.

LLoCa (Lorentz Local Canonicalization) makes a standard transformer Lorentz-equivariant
by predicting a local reference frame for each particle with an equivariant "frames-net",
expressing all features in these local frames, and transforming tensorial attention heads
between frames during message exchange.

Papers:
 - "Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant"
   https://arxiv.org/abs/2505.20280
 - "A Guide to Optimal Transformer Architectures for Jet Tagging"
   https://arxiv.org/abs/2608.02735 (tagging-guide, LLoCa-Transformer reference setup)

Ported from the `lloca` package, dev branch commit 8a5bb43
(https://github.com/heidelberg-hepml/lloca: ``lloca/backbone/transformer_v2.py``,
``lloca/backbone/attention.py``, ``lloca/framesnet/{frames,equi_frames}.py``,
``lloca/equivectors/mlp.py``, ``lloca/reps/*`` and ``lloca/utils/*``) and the tagging
wrapper of https://github.com/heidelberg-hepml/tagging-guide
(``experiments/tagging/wrappers.py`` and ``experiments/tagging/embedding.py``).

Differences with respect to the upstream implementation:
 - Everything runs on the dense zero-padded (batch, particles, channels) layout used by
   weaver; the equivectors edge convolution is a masked dense reimplementation of the
   upstream torch_geometric ``MessagePassing`` module (mathematically equivalent on the
   valid tokens), so there is no torch_geometric dependency.
 - The default attention backend is the native
   ``torch.nn.functional.scaled_dot_product_attention`` (boolean key-padding mask);
   ``attention_backend="varlen"``, ``"flash"``, or ``"xformers"`` instead drop the
   padding and run block-diagonal attention over the packed tokens (see
   :class:`weaver.nn.model.LGATrSlim` for the backend implementations). The frames-net
   stays dense in all cases.
 - The edge-attribute standardization of the frames-net is initialized lazily from the
   first batch (as in the upstream PELICAN-lite wrapper) instead of via an external
   ``init_standardization`` hook; the resulting statistics are stored in buffers and
   travel with the checkpoint.
 - ONNX export works but requires the float32 paths (``momentum_float64=False`` and
   ``ortho_use_float64=False``): onnxruntime lacks float64 kernels for some of the
   involved ops (e.g. Atan and the optimizer's FusedMatMul).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from torch import nn
from torch.utils.checkpoint import checkpoint

from weaver.nn.model.LGATrSlim import (
    ATTENTION_BACKENDS,
    _movedim,
    dense_to_sparse,
    get_sparse_attention_kwargs,
    get_spurion,
    insert_global_tokens,
    minimum_autocast_precision,
    scaled_dot_product_attention,
)
from weaver.nn.model.kinematics import (
    get_auxiliary_scalars,
    get_num_auxiliary_scalars,
)
from weaver.nn.model.ParticleTransformer import SequenceTrimmer
from weaver.utils.logger import _logger


# ------------------------------------------------------------------------------------
# Minkowski-space basics (ported from lloca/utils/lorentz.py)
# ------------------------------------------------------------------------------------


def lorentz_inner(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product v1^T @ g @ v2 for (..., 4) tensors in (E, px, py, pz)."""
    t = v1[..., 0] * v2[..., 0]
    s = (v1[..., 1:] * v2[..., 1:]).sum(dim=-1)
    return t - s


def lorentz_squarednorm(v: torch.Tensor) -> torch.Tensor:
    """Lorentz squared norm v^T @ g @ v for a (..., 4) tensor."""
    return lorentz_inner(v, v)


def lorentz_eye(dims, device=None, dtype=torch.float32) -> torch.Tensor:
    """Identity matrix expanded to shape (*dims, 4, 4)."""
    if device is None:
        device = torch.device("cpu")
    base_eye = torch.eye(4, dtype=dtype, device=device)
    return base_eye.view((1,) * len(dims) + (4, 4)).expand(*dims, 4, 4)



# ------------------------------------------------------------------------------------
# Orthogonalization (ported from lloca/utils/{orthogonalize_3d,orthogonalize_4d}.py)
# ------------------------------------------------------------------------------------


def regularize_collinear(vecs, eps_reg=None):
    """Add noise to (nearly) collinear 3-vector pairs so orthogonalization stays sound."""
    eps_reg = torch.finfo(vecs.dtype).eps if eps_reg is None else eps_reg

    v0, v1 = vecs.unbind(dim=-2)
    cross = torch.cross(v0, v1, dim=-1)
    mask = (cross**2).sum(dim=-1) < eps_reg
    v0_reg = torch.where(mask.unsqueeze(-1), v0 + eps_reg * torch.randn_like(v0), v0)
    v1_reg = torch.where(mask.unsqueeze(-1), v1 + eps_reg * torch.randn_like(v1), v1)
    vecs_reg = torch.stack([v0_reg, v1_reg], dim=-2)

    reg_collinear = mask.sum()
    return vecs_reg, reg_collinear


def orthogonalize_gramschmidt_3d(vecs, eps_norm=None):
    """Gram-Schmidt orthogonalization of two euclidean (..., 2, 3) vectors -> (..., 3, 3)."""
    vecs = F.normalize(vecs, dim=-1, eps=eps_norm)
    e0, v1 = vecs.unbind(dim=-2)

    u1 = v1 - (v1 * e0).sum(dim=-1, keepdim=True) * e0
    e1 = F.normalize(u1, dim=-1, eps=eps_norm)

    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


def orthogonalize_cross_3d(vecs, eps_norm=None):
    """Cross-product orthogonalization of two euclidean (..., 2, 3) vectors -> (..., 3, 3)."""
    vecs = F.normalize(vecs, dim=-1, eps=eps_norm)
    e0, v1 = vecs.unbind(dim=-2)

    u1 = torch.cross(e0, v1, dim=-1)
    e1 = F.normalize(u1, dim=-1, eps=eps_norm)

    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


def orthogonalize_3d(vecs, method="gramschmidt", eps_norm=None, eps_reg=None, return_reg=False):
    """Orthogonalize two euclidean vectors into a rotation matrix (see lloca)."""
    eps_norm = torch.finfo(vecs.dtype).eps if eps_norm is None else eps_norm

    vecs, reg_collinear = regularize_collinear(vecs, eps_reg)

    if method == "cross":
        trafo = orthogonalize_cross_3d(vecs, eps_norm)
    elif method == "gramschmidt":
        trafo = orthogonalize_gramschmidt_3d(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")

    return (trafo, reg_collinear) if return_reg else trafo


def regularize_lightlike(vecs, eps_reg_lightlike=None):
    """Add timelike noise to (nearly) lightlike Minkowski vectors."""
    eps_reg_lightlike = (
        torch.finfo(vecs.dtype).eps if eps_reg_lightlike is None else eps_reg_lightlike
    )
    inners = lorentz_squarednorm(vecs)
    mask = inners.abs() < eps_reg_lightlike

    # calculate the 3-norm and set the 0th-component accordingly
    randn_vecs = torch.randn_like(vecs).abs()
    randn_vecs_3sqnorm = (randn_vecs[..., 1:] ** 2).sum(dim=-1)
    randn_vecs[..., 0] = (2 * randn_vecs_3sqnorm).sqrt()  # heuristic factor of 2 to ensure timelike

    vecs_reg = vecs + mask.unsqueeze(-1) * eps_reg_lightlike * randn_vecs
    reg_lightlike = mask.any(dim=-1).sum() if vecs.dim() > 1 else mask.sum()
    return vecs_reg, reg_lightlike


# ------------------------------------------------------------------------------------
# Polar decomposition (ported from lloca/utils/polar_decomposition.py)
# ------------------------------------------------------------------------------------


def restframe_boost(fourmomenta, checks=False):
    """Lorentz transformation that boosts (..., 4) four-momenta into their rest frame."""
    if checks:
        assert (lorentz_squarednorm(fourmomenta) > 0).all(), (
            "Trying to boost spacelike vectors into their restframe (not possible). "
            "Consider changing the nonlinearity in equivectors."
        )

    # compute relevant quantities
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=1e-10)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    # prepare rotation part
    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    # collect trafo
    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    trafo = torch.cat((row0.unsqueeze(-2), lower), dim=-2)
    return trafo


def polar_decomposition(
    fourmomenta,
    references,
    use_float64=True,
    return_reg=False,
    eps_reg_lightlike=None,
    checks=False,
    **kwargs,
):
    """Construct a Lorentz transformation as a polar decomposition of a boost and a rotation.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4): the four-momenta that define the rest frames.
    references : torch.Tensor
        Tensor of shape (..., 2, 4): the reference four-momenta for the rotation.
    use_float64 / return_reg / eps_reg_lightlike / checks / **kwargs
        See lloca; ``kwargs`` are forwarded to :func:`orthogonalize_3d`.
    """
    assert fourmomenta.shape[:-1] == references.shape[:-2]

    if use_float64:
        original_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
        references = references.to(torch.float64)

    # fourmomenta for boost must be timelike
    fourmomenta, reg_lightlike = regularize_lightlike(fourmomenta, eps_reg_lightlike)

    # construct rest frame transformation
    boost = restframe_boost(fourmomenta, checks=checks)

    # references go into rest frame
    ref_rest = torch.matmul(references, boost.transpose(-1, -2))

    # construct rotation before orthogonalization
    ref3_rest = ref_rest[..., 1:]
    out = orthogonalize_3d(ref3_rest, return_reg=return_reg, **kwargs)
    if return_reg:
        orthogonal_vec3, reg_collinear = out
    else:
        orthogonal_vec3 = out
    rotation = torch.zeros_like(boost)
    rotation[..., 0, 0] = 1
    rotation[..., 1:, 1:] = orthogonal_vec3

    # combine rotation and boost
    trafo = torch.matmul(rotation, boost)
    if use_float64:
        trafo = trafo.to(original_dtype)
    return (trafo, reg_lightlike, reg_collinear) if return_reg else trafo


def soft_clamp(x, max=None, min=None, hardness=None):
    if hardness is None:
        # hard clamp
        return x.clamp(min=min, max=max)
    else:
        # soft clamp (better gradients)
        out = max - F.softplus(max - x, beta=hardness)
        return out.clamp(min=min)


def clamp_boost(x, gamma_max, gamma_hardness):
    """Optionally clamp the gamma factor of predicted boost vectors (see lloca equi_frames)."""
    mass = lorentz_squarednorm(x).clamp(min=0).sqrt().unsqueeze(-1)
    t0 = x.narrow(-1, 0, 1)
    beta = x[..., 1:] / t0.clamp_min(1e-10)
    gamma = t0 / mass
    gamma_max_realized = gamma.max().detach()
    gamma_mean = gamma.mean().detach()

    if gamma_max is None:
        return x, None, gamma_mean, gamma_max_realized

    else:
        # carefully clamp gamma to keep boosts under control
        reg_gammamax = (gamma > gamma_max).sum().detach()
        gamma_reg = soft_clamp(gamma, min=1, max=gamma_max, hardness=gamma_hardness)
        beta_scaling = (
            torch.sqrt(torch.clamp(1 - 1 / gamma_reg.clamp(min=1e-10).square(), min=1e-10))
            / (beta**2).sum(dim=-1, keepdim=True).clamp(min=1e-10).sqrt()
        )
        beta_reg = beta * beta_scaling
        x_reg = mass * torch.cat((gamma_reg, gamma_reg * beta_reg), dim=-1)
        return x_reg, reg_gammamax, gamma_mean, gamma_max_realized


# ------------------------------------------------------------------------------------
# Frames bookkeeping (ported from lloca/framesnet/frames.py)
# ------------------------------------------------------------------------------------


class Frames:
    """Bookkeeping class for local frames.

    Collection of Lorentz transformations, represented as (..., 4, 4) matrices.
    Properties like det and inv are cached for performance. Attributes should not
    be changed after initialization to avoid inconsistencies.
    """

    def __init__(
        self,
        matrices: torch.Tensor = None,
        is_global: bool = False,
        det: torch.Tensor = None,
        inv: torch.Tensor = None,
        is_identity: bool = False,
        shape=None,
        device=None,
        dtype: torch.dtype = None,
    ):
        # straight-forward initialization
        self.is_identity = is_identity
        if is_identity:
            if matrices is None:
                assert shape is not None and device is not None and dtype is not None
            else:
                shape = matrices.shape[:-2]
                device = matrices.device
                dtype = matrices.dtype

            self.matrices = lorentz_eye(shape, device=device, dtype=dtype)
            self.is_global = True
            self.det = torch.ones(self.shape[:-2], dtype=self.dtype, device=self.device)
            self.inv = self.matrices
        else:
            assert matrices is not None
            assert matrices.shape[-2:] == (4, 4), (
                f"matrices must be of shape (..., 4, 4), but found {matrices.shape[-2:]} instead"
            )

            self.matrices = matrices
            self.is_global = is_global
            if det is not None:
                assert det.shape == matrices.shape[:-2]
            if inv is not None:
                assert inv.shape == matrices.shape
            self.det = det
            self.inv = inv

        # cache expensive properties
        if self.det is None:
            self.det = torch.linalg.det(self.matrices)
        if self.inv is None:
            self.inv = self.matrices.transpose(-1, -2).clone()
            self.inv[..., 1:, :] *= -1
            self.inv[..., :, 1:] *= -1

    def reshape(self, *shape):
        assert shape[-2:] == (4, 4)
        return Frames(
            matrices=self.matrices.reshape(*shape),
            is_identity=self.is_identity,
            is_global=self.is_global,
            inv=self.inv.reshape(*shape),
            det=self.det.reshape(*shape[:-2]),
        )

    def expand(self, *shape):
        assert shape[-2:] == (4, 4)
        return Frames(
            matrices=self.matrices.expand(*shape),
            is_identity=self.is_identity,
            is_global=self.is_global,
            inv=self.inv.expand(*shape),
            det=self.det.expand(*shape[:-2]),
        )

    def to(self, dtype=None, device=None):
        """Move the matrices to a new device and/or dtype (in place)."""
        self.matrices = self.matrices.to(device=device, dtype=dtype)
        self.inv = self.inv.to(device=device, dtype=dtype)
        self.det = self.det.to(device=device, dtype=dtype)

    def __repr__(self):
        return repr(self.matrices)

    @property
    def device(self):
        return self.matrices.device

    @property
    def dtype(self):
        return self.matrices.dtype

    @property
    def shape(self):
        return self.matrices.shape


class InverseFrames(Frames):
    """Inverse of a collection of frames."""

    def __init__(self, frames: Frames):
        super().__init__(
            matrices=frames.inv,
            is_global=frames.is_global,
            inv=frames.matrices,
            det=frames.det,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.shape,
        )


class LowerIndicesFrames(Frames):
    """Frames with lower indices, obtained by multiplying with the metric.

    Used in LLoCaAttention to lower the key indices.
    """

    def __init__(self, frames):
        matrices = frames.matrices.clone()
        matrices[..., 1:, :] *= -1
        inv = frames.inv.clone()
        inv[..., :, 1:] *= -1
        det = -frames.det

        super().__init__(
            matrices=matrices,
            inv=inv,
            det=det,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.shape,
        )


# ------------------------------------------------------------------------------------
# Lorentz tensor representations (ported from lloca/reps/tensorreps.py)
# ------------------------------------------------------------------------------------


class TensorRep(tuple):
    """Individual tensor representation, identified by its order and parity."""

    def __new__(cls, order, parity):
        assert isinstance(order, int) and order >= 0, (
            f"order must be a non-negative integer, but got {order}"
        )
        assert parity in [-1, 1], f"parity must be either -1 (p) or 1 (n), but got {parity}"
        return super().__new__(cls, (order, parity))

    def __deepcopy__(self, memo):
        return self

    @property
    def order(self):
        return self[0]

    @property
    def parity(self):
        return self[1]

    def __repr__(self):
        return f"{self.order}{'n' if self.parity == 1 else 'p'}"


class _TensorMulRep(tuple):
    """Direct product of similar tensor representations."""

    def __new__(cls, mul, rep):
        assert isinstance(mul, int) and mul >= 0, (
            f"mul must be a non-negative integer, but got {mul}"
        )
        assert isinstance(rep, TensorRep), (
            f"rep must be an instance of TensorRep, but got type {type(rep)}"
        )
        return super().__new__(cls, (mul, rep))

    def __deepcopy__(self, memo):
        return self

    @property
    def mul(self):
        return self[0]

    @property
    def rep(self):
        return self[1]

    @property
    def dim(self):
        return (4 ** self.rep.order) * self.mul

    def __repr__(self):
        return f"{self.mul}x{self.rep}"


class TensorReps(tuple):
    """Generic tensor representations, e.g. ``TensorReps("8x0n+2x1n")``."""

    def __new__(cls, input, simplify=True):
        if isinstance(input, TensorReps):
            tensor_reps = input
        elif isinstance(input, (list, tuple)):
            assert all(isinstance(x, _TensorMulRep) for x in input)
            tensor_reps = input
        elif isinstance(input, str):
            try:
                tensor_reps = parse_tensorreps_string(input)
            except ValueError as err:
                raise ValueError(f"Invalid tensor_reps string {input}") from err
        else:
            raise ValueError(f"Invalid input: {input} is of type {type(input)}")

        ret = super().__new__(cls, tensor_reps)
        if simplify:
            return ret.simplify()
        return ret

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def __deepcopy__(self, memo):
        return self

    @property
    def dim(self):
        return sum(mul_ir.dim for mul_ir in self)

    @property
    def max_rep(self):
        return max(self, key=lambda x: x.rep.order)

    @property
    def is_sorted(self):
        if len(self) <= 1:
            return True
        return all(self[i].rep.order <= self[i + 1].rep.order for i in range(len(self) - 1))

    def sort(self):
        return TensorReps(sorted(self, key=lambda x: x.rep.order))

    def simplify(self):
        items = self if self.is_sorted else self.sort()

        out = []
        for mul_rep in items:
            mul, rep = mul_rep
            if len(out) > 0 and out[-1].rep == rep:
                # same rep -> extend mul of previous rep
                out[-1] = _TensorMulRep(out[-1].mul + mul, rep)
            elif mul > 0:
                # different rep and mul>0 -> create new entry
                out.append(mul_rep)

        return TensorReps(out, simplify=False)


def parse_tensorreps_string(input):
    """Parse e.g. "2x2n+3x1p+1x0n" into a list of _TensorMulRep instances."""
    out = []
    input = input.replace(" ", "")  # remove whitespace
    input_list = input.split("+")  # split into single _TensorMulRep
    for rep in input_list:
        if rep[-1] == "n":
            parity = 1
            rep = rep[:-1]
        elif rep[-1] == "p":
            parity = -1
            rep = rep[:-1]
        else:
            raise ValueError(
                f"Invalid last character (=parity) in tensorreps string {rep}, "
                "should be either 'n' or 'p'"
            )

        mul, order = rep.split("x")
        mul, order = int(mul), int(order)
        out.append(_TensorMulRep(mul, TensorRep(order, parity)))
    return out


# ------------------------------------------------------------------------------------
# Tensor representation transform (ported from lloca/reps/tensorreps_transform.py)
# ------------------------------------------------------------------------------------


class TensorRepsTransform(nn.Module):
    """Transform a tensor of a given representation with per-item local frames."""

    def __init__(self, reps: TensorReps, use_naive=False):
        super().__init__()
        self.reps = reps
        self.transform = self._transform_naive if use_naive else self._transform_efficient

        # cache idx_start and idx_end for each rep
        self.start_end_idx = []
        idx = 0
        for mul_rep in self.reps:
            self.start_end_idx.append([idx, idx + mul_rep.dim])
            idx += mul_rep.dim

        # build parity_mask
        parity_odd = torch.zeros(self.reps.dim, dtype=torch.bool)
        idx = 0
        for mul_rep in self.reps:
            _, rep = mul_rep
            parity_odd[idx: idx + mul_rep.dim] = True if rep.parity == -1 else False
            idx += mul_rep.dim
        self.register_buffer("parity_odd", parity_odd.unsqueeze(0))
        self.no_parity_odd = parity_odd.sum().item() == 0

        if not use_naive:
            # build mapping from order to element in the reps list
            self.map_rep = [None for _ in range(self.reps.max_rep.rep.order + 1)]
            idx_rep = 0
            for i in range(self.reps.max_rep.rep.order + 1):
                if self.reps[idx_rep].rep.order == i:
                    self.map_rep[i] = idx_rep
                    idx_rep += 1

        if self.reps.max_rep.rep.order <= 1:
            # super efficient shortcut if only scalar and vector reps are present
            self.transform = self._transform_only_scalars_and_vectors

        self.has_higher_orders = self.reps.max_rep.rep.order > 0

    @minimum_autocast_precision(torch.float32, output=None)
    def forward(self, tensor: torch.Tensor, frames: Frames) -> torch.Tensor:
        """Apply the frame transformation to a (..., reps.dim) tensor."""
        if frames.is_identity or (self.no_parity_odd and not self.has_higher_orders):
            return tensor

        in_shape = tensor.shape
        if len(frames.shape) > 3:
            frames = frames.reshape(-1, 4, 4)
        tensor = tensor.reshape(-1, tensor.shape[-1])
        assert tensor.shape[0] == frames.shape[0], (
            f"Batch dimension is {tensor.shape[0]} for tensor, but {frames.shape[0]} for frames."
        )

        tensor_transformed = self.transform(tensor, frames) if self.has_higher_orders else tensor
        tensor_transformed = self.transform_parity(tensor_transformed, frames)

        return tensor_transformed.view(*in_shape)

    def _transform_naive(self, tensor, frames):
        """Apply n transformations to each tensor of n'th order, rep by rep."""
        output = tensor.clone()
        frames = frames.matrices.clone().to(tensor.dtype)
        for mul_rep, (idx_start, idx_end) in zip(self.reps, self.start_end_idx):
            mul, rep = mul_rep
            if mul == 0 or rep.order == 0:
                continue

            x = tensor[:, idx_start:idx_end].reshape(-1, mul, *([4] * rep.order))

            einsum_string = get_einsum_string(rep.order)
            x_transformed = torch.einsum(einsum_string, *([frames] * rep.order), x)
            output[:, idx_start:idx_end] = x_transformed.reshape(-1, mul_rep.dim)

        return output

    def _transform_efficient(self, tensor, frames):
        """Transform starting with the highest order, merging lower orders on the way down."""
        output = None
        bframes = frames.matrices.clone().to(tensor.dtype)
        for order in reversed(range(self.reps.max_rep.rep.order + 1)):
            if self.map_rep[order] is not None:
                # add new contribution to the mix
                idx_start, idx_end = self.start_end_idx[self.map_rep[order]]
                contribution = tensor[:, idx_start:idx_end].reshape(
                    tensor.shape[0], -1, *(order * (4,))
                )
                output = (
                    torch.cat([contribution, output], dim=1)
                    if order < self.reps.max_rep.rep.order
                    else contribution
                )

            if order > 0:
                # apply transformation, then flatten because transformation is done
                output = torch.einsum("ijk,ilk...->ilj...", bframes, output)
                output = output.flatten(start_dim=1, end_dim=2)

        return output

    def _transform_only_scalars_and_vectors(self, tensor, frames):
        """Shortcut that assumes only scalar and vector reps are present."""
        N, D = tensor.shape
        vec_start, vec_end = self.start_end_idx[-1]
        vec_width = vec_end - vec_start
        L = vec_width // 4
        vectors = tensor.narrow(1, vec_start, vec_width).view(N, L, 4)
        mats = frames.matrices
        if mats.dtype != vectors.dtype:
            mats = mats.to(vectors.dtype)

        out_vecs = (mats.unsqueeze(1) * vectors.unsqueeze(-2)).sum(-1)

        if vec_start == 0 and vec_end == D:
            out = out_vecs.reshape(N, D)
        else:
            out = torch.empty_like(tensor)
            if vec_start > 0:
                out[:, :vec_start] = tensor[:, :vec_start]
            out[:, vec_start:vec_end] = out_vecs.reshape(N, vec_width)
        return out

    def transform_parity(self, tensor, frames):
        """Multiply parity-odd states by sign(det Lambda)."""
        if self.no_parity_odd:
            return tensor
        return torch.where(self.parity_odd, frames.det.sign().unsqueeze(-1) * tensor, tensor)


def get_einsum_string(order):
    """Create the einsum string for the naive transformation of an order-n tensor."""
    if order > 12:
        raise NotImplementedError("Running out of letters for order>12")

    einsum = ""
    start = ord("A")
    batch_index = ord("a")

    # list of frames
    for i in range(order):
        einsum += chr(batch_index) + chr(start + 2 * i) + chr(start + 2 * i + 1) + ","

    # tensor
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i + 1)

    # output
    einsum += "->"
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i)

    return einsum


# ------------------------------------------------------------------------------------
# LLoCa attention (ported from lloca/backbone/attention.py, dense layout only)
# ------------------------------------------------------------------------------------


def _scale_frames(frames: Frames, scale: torch.Tensor) -> Frames:
    """Uniformly rescale each frame matrix by a per-particle scalar factor.

    A grade-n tensor transform applies the frame matrix n times, so scaling the matrices by
    ``scale`` rescales grade-n channels by ``scale**n`` (grade 0 / scalars are untouched) --
    the same effect as dividing the post-transform tensor by a per-channel gamma**grade
    divisor, but folded once into the (much smaller) frame matrices instead of applied to
    every q/k/v/output tensor in every layer.
    """
    return Frames(
        matrices=frames.matrices * scale,
        is_global=frames.is_global,
        inv=frames.inv / scale,
        det=frames.det * scale[..., 0, 0] ** 4,
    )


class LLoCaAttention(nn.Module):
    """Attention with frame-to-frame transformations.

    Parameters
    ----------
    attn_reps : TensorReps
        Tensor representation of a single attention head.
    num_heads : int
        Number of attention heads.
    preserve_variance : bool
        Rescale the pre-attention (local->global) q/k/v and post-attention (global->local)
        vectors by 1/gamma_i^grade to prevent the variance blowup from large boosts. Needs
        the reference momentum ``p_ref`` in :meth:`prepare_frames`.
    variance_eps : float
        Small mass floor (energy units) that keeps gamma_i finite for near-lightlike jets.
    """

    def __init__(self, attn_reps, num_heads, preserve_variance=True, variance_eps=1e-2):
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))
        self.num_heads = num_heads
        self.preserve_variance = preserve_variance
        self.variance_eps = variance_eps

        self.frames = None
        self.frames_qkv = None
        self.frames_out = None

    def _compute_gamma(self, frames, p_ref):
        """Invariant per-particle Lorentz factor gamma_i >= 1 that prevents variance blowup."""
        dtype = torch.promote_types(p_ref.dtype, torch.float32)
        L = frames.matrices.to(dtype)
        p_ref = p_ref.to(dtype)
        if p_ref.shape[:-1] != L.shape[:-2]:
            # dense layout: one reference momentum per event, broadcast over the token
            # axis; the packed layout passes token-resolved reference momenta instead
            p_ref = p_ref.unsqueeze(-2).expand(*L.shape[:-2], 4)
        m_ref = torch.sqrt(self.variance_eps**2 + lorentz_squarednorm(p_ref).clamp(min=0))
        # only row 0 of (L @ p_ref) is needed: gamma = (Lambda p_ref)^0 / m_ref
        gamma = (L[..., 0, :] * p_ref).sum(dim=-1) / m_ref
        return gamma.detach()  # fixed normalization: no gradient into the frames

    @minimum_autocast_precision(torch.float32, output=None)
    def prepare_frames(self, frames, p_ref=None):
        """Prepare local frames for LLoCa attention (called once per forward pass).

        Parameters
        ----------
        frames: Frames
            Local frames of shape (..., N, 4, 4).
        p_ref: torch.Tensor, optional
            Reference 4-momentum in the global frame (energy-first), i.e. the total (jet)
            momentum per event, shape (..., 4). Required when ``preserve_variance`` is on.
        """
        self.frames = frames
        if not frames.is_global:
            inv_gamma = None
            if self.preserve_variance:
                if p_ref is None:
                    raise ValueError("preserve_variance requires `p_ref` in prepare_frames.")
                gamma = self._compute_gamma(frames, p_ref)
                # (..., 1, N, 1, 1): broadcasts over heads and the 4x4 matrix
                inv_gamma = (1 / gamma)[..., None, :, None, None]

            # insert frames head dimension
            frames_out = frames.reshape(*frames.shape[:-3], 1, frames.shape[-3], 4, 4)
            frames_out = frames_out.expand(
                *frames.shape[:-3], self.num_heads, frames.shape[-3], 4, 4
            )

            # create inv_frames and lower_inv_frames
            inv_frames = InverseFrames(frames_out)
            lower_inv_frames = LowerIndicesFrames(inv_frames)

            if self.preserve_variance:
                # rescale the pre-attention (local->global) q/k/v transform
                inv_frames = _scale_frames(inv_frames, inv_gamma)
                lower_inv_frames = _scale_frames(lower_inv_frames, inv_gamma)

            # qkv = (inv_frames, lower_inv_frames, inv_frames)
            # note that (lower_inv_frames, inv_frames, inv_frames) is equivalent
            self.frames_qkv = Frames(
                matrices=torch.cat(
                    [
                        inv_frames.matrices,
                        lower_inv_frames.matrices,
                        inv_frames.matrices,
                    ],
                    dim=0,
                ),
                is_identity=inv_frames.is_identity,
                is_global=inv_frames.is_global,
                det=torch.cat([inv_frames.det, lower_inv_frames.det, inv_frames.det], dim=0),
                inv=torch.cat([inv_frames.inv, lower_inv_frames.inv, inv_frames.inv], dim=0),
            )

            if self.preserve_variance:
                # rescale the post-attention (global->local) output transform
                frames_out = _scale_frames(frames_out, inv_gamma)

            # flatten frames (preparation for tensorreps_transform)
            self.frames_out = frames_out.reshape(-1, 4, 4)
            self.frames_qkv = self.frames_qkv.reshape(-1, 4, 4)

    def _local_to_global(self, q_local, k_local, v_local):
        # check input shapes
        assert k_local.shape == v_local.shape == q_local.shape  # has to match perfectly
        assert 3 * q_local.shape[:-1].numel() == self.frames_qkv.shape[-3]

        # transform q, k, v into global frame (preserve_variance rescaling, if enabled, is
        # already folded into self.frames_qkv, see prepare_frames)
        qkv_local = torch.cat([q_local, k_local, v_local], dim=0)
        qkv_global = self.transform(qkv_local, self.frames_qkv)
        q_global, k_global, v_global = qkv_global.chunk(3, dim=0)
        return q_global, k_global, v_global

    def _global_to_local(self, out_global):
        # transform result back into local frame (preserve_variance rescaling, if enabled,
        # is already folded into self.frames_out, see prepare_frames)
        return self.transform(out_global, self.frames_out)

    def forward(self, q_local, k_local, v_local, **attn_kwargs):
        """Execute LLoCa attention on (..., H, N, C) local queries/keys/values.

        Strategy: transform q, k, v into the global frame, apply attention there,
        and transform the output back into the local frames.
        """
        if self.frames.is_global:
            # fallback to standard attention for global frames
            return scaled_dot_product_attention(q_local, k_local, v_local, **attn_kwargs)

        q_global, k_global, v_global = self._local_to_global(q_local, k_local, v_local)

        # (B, H, N, C) format required by the attention backends
        shape_q, shape_k = q_global.shape, k_global.shape
        q_global = q_global.reshape(-1, *shape_q[-3:])
        k_global = k_global.reshape(-1, *shape_k[-3:])
        v_global = v_global.reshape(-1, *shape_k[-3:])

        # attention (in global frame); backend selected by the attention kwargs
        out_global = scaled_dot_product_attention(q_global, k_global, v_global, **attn_kwargs)

        out_global = out_global.view(*shape_q)  # (..., H, N, C)

        out_local = self._global_to_local(out_global)
        return out_local


# ------------------------------------------------------------------------------------
# Transformer backbone (ported from lloca/backbone/transformer_v2.py)
# ------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Drop-in replacement for :class:`torch.nn.RMSNorm`.

    The TorchScript ONNX exporter does not support ``aten::rms_norm``, so the
    normalization is spelled out; the parameter layout matches ``nn.RMSNorm``.
    """

    def __init__(self, normalized_shape: int, elementwise_affine: bool = True,
                 eps: float | None = None):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values for multi-head attention."""

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels)

    def forward(self, inputs):
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels)

        *leading, items, last = qkv.shape
        hidden_channels = last // (3 * self.num_heads)
        qkv = qkv.view(*leading, items, 3, hidden_channels, self.num_heads)
        qkv = _movedim(_movedim(qkv, -3, 0), -1, len(leading) + 1)
        q, k, v = qkv.unbind(dim=0)  # 3x (..., num_heads, num_items, hidden_channels // num_heads)
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer wrapping an LLoCaAttention instance."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        attention,
        num_heads: int = 8,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.attention = attention

        # Linear maps
        self.qkv_linear = MultiHeadQKVLinear(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels, out_channels)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        q, k, v = self.qkv_linear(inputs)  # each: (..., num_heads, num_items, num_channels)

        # Attention layer
        h = self.attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            **attn_kwargs,
        )

        # Concatenate heads and transform linearly
        # (permute with normalized dims: negative dims can produce invalid Transpose
        # nodes in the TorchScript ONNX exporter)
        *leading, num_heads, num_items, channels_per_head = h.shape
        n = h.dim()
        h = h.permute(*range(n - 3), n - 2, n - 3, n - 1)
        h = h.reshape(*leading, num_items, num_heads * channels_per_head)

        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block: pre-norm attention and pre-norm gated MLP with residuals."""

    def __init__(
        self,
        hidden_channels,
        attention,
        num_heads: int = 8,
        attention_factor: int = 1,
        mlp_factor: int = 2,
        dropout_prob=None,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(
            normalized_shape=hidden_channels, elementwise_affine=elementwise_affine
        )
        self.norm2 = RMSNorm(
            normalized_shape=hidden_channels, elementwise_affine=elementwise_affine
        )

        hidden_channels_attn = hidden_channels * attention_factor

        self.attention = BaselineSelfAttention(
            hidden_channels,
            hidden_channels,
            hidden_channels_attn,
            attention,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )

        self.mlp_in = nn.Sequential(
            nn.Linear(hidden_channels, 2 * mlp_factor * hidden_channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(mlp_factor * hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        # Residual attention
        h = self.norm1(inputs).to(inputs.dtype)
        h = self.attention(h, **attn_kwargs)
        outputs = inputs + h

        # Residual MLP with GatedLinearUnit
        h = self.norm2(outputs).to(outputs.dtype)
        h1, h2 = self.mlp_in(h).chunk(2, dim=-1)
        h = self.act(h1) * h2
        h = self.mlp_out(h)
        outputs = outputs + h

        return outputs


class LLoCaTransformer(nn.Module):
    """LLoCa-Transformer backbone (``lloca.backbone.transformer_v2.Transformer``).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    attn_reps : str
        Tensor representation of each attention head, e.g. ``"8x0n+2x1n"``.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    checkpoint_blocks : bool
        Use gradient checkpointing for transformer blocks.
    attention_factor : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    mlp_factor : int
        Factor by which the activation size is increased over the default value of
        hidden_channels.
    dropout_prob : float
        Dropout probability for output.
    preserve_variance : bool
        Rescale tensorial attention channels with the per-particle gamma factor to prevent
        variance blowup from large boosts (see :class:`LLoCaAttention`).
    elementwise_affine : bool
        Whether the RMSNorm layers use learnable per-channel affine weights.
    compile : bool
        Whether to wrap the forward with :func:`torch.compile`.
    compile_kwargs : dict, optional
        Forwarded verbatim to :func:`torch.compile` when ``compile=True``.
    """

    def __init__(
        self,
        in_channels: int,
        attn_reps: str,
        out_channels: int,
        num_blocks: int,
        num_heads: int,
        checkpoint_blocks: bool = False,
        attention_factor: int = 1,
        mlp_factor: int = 2,
        dropout_prob: float | None = None,
        preserve_variance: bool = True,
        elementwise_affine: bool = True,
        compile: bool = False,
        compile_kwargs=None,
    ) -> None:
        super().__init__()
        attn_reps = TensorReps(attn_reps)
        self.hidden_channels = attn_reps.dim * num_heads // attention_factor
        self.checkpoint_blocks = checkpoint_blocks
        self.attention = LLoCaAttention(
            attn_reps,
            num_heads,
            preserve_variance=preserve_variance,
        )

        self.linear_in = nn.Linear(in_channels, self.hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    self.hidden_channels,
                    attention=self.attention,
                    num_heads=num_heads,
                    attention_factor=attention_factor,
                    mlp_factor=mlp_factor,
                    dropout_prob=dropout_prob,
                    elementwise_affine=elementwise_affine,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(self.hidden_channels, out_channels)

        if compile:
            # rebind self.forward rather than patching the class to keep compilation instance-local
            self.forward = torch.compile(self.forward, **dict(compile_kwargs or {}))

    def forward(self, inputs: torch.Tensor, frames, p_ref=None, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_items, in_channels).
        frames : Frames
            Local frames used for invariant particle attention.
        p_ref : Tensor, optional
            Reference (jet) 4-momentum in the global frame, energy-first, shape (..., 4).
            Required when ``preserve_variance`` is on, ignored otherwise.
        **attn_kwargs
            Forwarded to attention (e.g. ``attn_mask``).
        """
        self.attention.prepare_frames(frames, p_ref=p_ref)

        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                fn = partial(block, **attn_kwargs)
                h = checkpoint(fn, h, use_reentrant=False)
            else:
                h = block(h, **attn_kwargs)
        outputs = self.linear_out(h)
        return outputs


# ------------------------------------------------------------------------------------
# Equivariant vector predictors (masked dense port of lloca/equivectors/mlp.py)
# ------------------------------------------------------------------------------------


class MLP(nn.Module):
    """A simple MLP with GELU nonlinearities (ported from lloca/backbone/mlp.py)."""

    def __init__(self, in_channels, out_channels, hidden_channels, hidden_layers,
                 dropout_prob=None):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        layers: list = [nn.Linear(in_channels, hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        return self.mlp(inputs)


class EquiEdgeConv(nn.Module):
    """Equivariant edge convolution, masked dense port of the upstream MessagePassing module.

    For each valid receiver particle i, an MLP predicts positive weights over the valid
    sender particles j != i (softmax by default), and the output vectors are the weighted
    sums of the (normalized) relative four-momenta:
    ``vecs_i = sum_j w_ij * (p_i op p_j) / |p_i op p_j|``.

    The choice of the parameters ``operation``, ``nonlinearity``, ``fm_norm``, ``layer_norm``
    is critical to the stability of the approach; the defaults worked for all upstream
    experiments.

    Parameters
    ----------
    out_vectors : int
        Number of output vectors per particle.
    num_scalars : int
        Number of scalar features per particle.
    hidden_channels : int
        Number of hidden channels in the MLP.
    num_layers_mlp : int
        Number of hidden layers in the MLP.
    include_edges : bool
        Whether to include (standardized) invariant-mass edge attributes in the MLP input.
    operation : str
        Operation on the fourmomenta pairs: "add", "diff", or "single".
    nonlinearity : str
        Nonlinearity for the MLP output: "exp", "softplus", or "softmax".
    fm_norm : bool
        Whether to normalize the relative fourmomentum.
    layer_norm : bool
        Whether to apply Lorentz-equivariant layer normalization to the output vectors.
    use_amp : bool
        Whether to run the MLP under autocast.
    dropout_prob : float
        Dropout probability for the MLP.
    """

    def __init__(
        self,
        out_vectors,
        num_scalars,
        hidden_channels,
        num_layers_mlp=2,
        include_edges=True,
        operation="add",
        nonlinearity="softmax",
        fm_norm=True,
        layer_norm=True,
        use_amp=False,
        dropout_prob=None,
    ):
        super().__init__()
        assert num_scalars > 0 or include_edges, (
            "Either num_scalars > 0 or include_edges==True, otherwise there are no inputs."
        )
        assert operation in ("add", "diff", "single"), f"Invalid operation {operation}"
        assert nonlinearity in ("exp", "softplus", "softmax"), (
            f"Invalid nonlinearity {nonlinearity}"
        )
        assert not (operation == "single" and fm_norm), (
            "The setup operation=single and fm_norm==True is unstable"
        )
        self.include_edges = include_edges
        self.layer_norm = layer_norm
        self.operation = operation
        self.nonlinearity = nonlinearity
        self.fm_norm = fm_norm
        self.use_amp = use_amp

        in_edges = 1 if include_edges else 0
        in_channels = 2 * num_scalars + in_edges
        self.mlp = MLP(
            in_channels=in_channels,
            out_channels=out_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    @staticmethod
    def _pair_mask(mask):
        # (B, N, N) boolean mask of valid sender/receiver pairs, self-loops removed
        # (arange comparison instead of torch.eye: the dynamic-size EyeLike ONNX op
        # has no boolean kernel in onnxruntime)
        idx = torch.arange(mask.size(1), device=mask.device)
        not_self = idx.unsqueeze(0) != idx.unsqueeze(1)
        return mask.unsqueeze(2) & mask.unsqueeze(1) & not_self

    @staticmethod
    def _get_edge_attr(fourmomenta, eps=1e-10, use_float64=True):
        """log((p_i + p_j)^2) for all pairs; shape (B, N, N)."""
        if use_float64:
            in_dtype = fourmomenta.dtype
            fourmomenta = fourmomenta.to(torch.float64)
        psum = fourmomenta.unsqueeze(2) + fourmomenta.unsqueeze(1)
        edge_attr = lorentz_squarednorm(psum).clamp(min=eps).log()
        if use_float64:
            edge_attr = edge_attr.to(in_dtype)
        return edge_attr

    def init_standardization(self, fourmomenta, mask):
        """Compute edge-attribute standardization statistics from a reference batch."""
        if self.include_edges and not self.edge_inited:
            pair_mask = self._pair_mask(mask)
            edge_attr = self._get_edge_attr(fourmomenta)
            vals = edge_attr[pair_mask]
            self.edge_mean.copy_(vals.mean().detach())
            self.edge_std.copy_(vals.std().clamp(min=1e-5).detach())
            if dist.is_available() and dist.is_initialized():
                # broadcast rank-0 stats so every rank normalizes identically
                dist.broadcast(self.edge_mean, src=0)
                dist.broadcast(self.edge_std, src=0)
            self.edge_inited.fill_(True)

    def _apply_nonlinearity(self, prefactor, pair_mask):
        mask_e = pair_mask.unsqueeze(-1)
        if self.nonlinearity == "exp":
            return prefactor.clamp(min=-10, max=10).exp() * mask_e
        elif self.nonlinearity == "softplus":
            return F.softplus(prefactor) * mask_e
        # softmax over the sender axis, restricted to valid pairs; mirrors the upstream
        # sparse softmax (detached per-segment max, +1e-16 in the denominator)
        logits = prefactor.masked_fill(~mask_e, float("-inf"))
        src_max = logits.amax(dim=2, keepdim=True).detach()
        src_max = torch.where(torch.isfinite(src_max), src_max, torch.zeros_like(src_max))
        num = (logits - src_max).exp()
        denom = num.sum(dim=2, keepdim=True) + 1e-16
        return num / denom

    def forward(self, fourmomenta, scalars, mask):
        """
        Parameters
        ----------
        fourmomenta : torch.Tensor
            Four-momenta of shape (B, N, 4) in (E, px, py, pz).
        scalars : torch.Tensor
            Scalar features of shape (B, N, num_scalars).
        mask : torch.BoolTensor
            Valid-particle mask of shape (B, N).

        Returns
        -------
        torch.Tensor
            Predicted vectors of shape (B, N, out_vectors, 4); zero for masked particles.
        """
        B, N = mask.shape
        pair_mask = self._pair_mask(mask)

        # calculate and standardize edge attributes
        if self.include_edges:
            if not self.edge_inited:
                # lazy initialization from the first batch (upstream initializes from the
                # first training batch through an external hook)
                self.init_standardization(fourmomenta, mask)
            edge_attr = (self._get_edge_attr(fourmomenta) - self.edge_mean) / self.edge_std
            # related to the momentum_float64 option
            edge_attr = edge_attr.to(scalars.dtype).unsqueeze(-1)

        # per-edge MLP weights
        s_i = scalars.unsqueeze(2).expand(B, N, N, scalars.size(-1))
        s_j = scalars.unsqueeze(1).expand(B, N, N, scalars.size(-1))
        prefactor = torch.cat([s_i, s_j], dim=-1)
        if self.include_edges:
            prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        with torch.autocast(prefactor.device.type, enabled=self.use_amp):
            prefactor = self.mlp(prefactor)
        weights = self._apply_nonlinearity(prefactor, pair_mask)  # (B, N, N, V)

        # relative four-momenta
        p_i = fourmomenta.unsqueeze(2)
        p_j = fourmomenta.unsqueeze(1)
        if self.operation == "add":
            fm_rel = p_i + p_j
        elif self.operation == "diff":
            fm_rel = p_i - p_j
        else:  # single
            fm_rel = p_j.expand(B, N, N, 4)
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel_norm = fm_rel_norm.abs().sqrt().clamp(min=1e-6)
            fm_rel = fm_rel / fm_rel_norm

        # aggregate over the sender axis
        weights = weights.to(torch.promote_types(weights.dtype, fm_rel.dtype))
        fm_rel = fm_rel.to(weights.dtype)
        vecs = torch.einsum("bijv,bijc->bivc", weights, fm_rel)  # (B, N, V, 4)

        # equivariant layer normalization
        if self.layer_norm:
            norm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True)
            vecs = vecs / norm.abs().sqrt().clamp(min=1e-5).unsqueeze(-1)
        return vecs


class MLPVectors(nn.Module):
    """Edge convolution with a simple MLP (dense port of ``lloca.equivectors.mlp.MLPVectors``)."""

    def __init__(self, n_vectors, *args, **kwargs):
        super().__init__()
        # only a single message-passing block is supported (see upstream comment)
        self.block = EquiEdgeConv(*args, out_vectors=n_vectors, **kwargs)

    def init_standardization(self, fourmomenta, mask):
        self.block.init_standardization(fourmomenta, mask)

    def forward(self, fourmomenta, scalars, mask):
        return self.block(fourmomenta, scalars=scalars, mask=mask)


# ------------------------------------------------------------------------------------
# Learned frames (masked dense port of lloca/framesnet/equi_frames.py, LearnedPDFrames)
# ------------------------------------------------------------------------------------


def _init_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class LearnedPDFrames(nn.Module):
    """Frames as learnable polar decompositions (the upstream default approach).

    An equivariant vector predictor proposes three four-vectors per particle; the first
    defines a rest-frame boost and the other two an orthonormalized rotation, combined
    into a proper Lorentz transformation per particle. Masked particles get identity
    frames.

    Parameters
    ----------
    equivectors : callable
        Factory ``equivectors(n_vectors=3)`` returning the vector-predictor module.
    is_global : bool
        If True, average the predicted vectors over each event to construct a global frame.
    random : bool
        If True, re-initialize the equivectors at each forward pass (data augmentation).
    fix_params : bool
        Freeze the frames-net parameters.
    mass_reg : float or None
        Lift particles below this mass onto the mass shell before predicting vectors.
    gamma_max : float or None
        Maximum gamma factor for boost regularization (None: no regularization).
    gamma_hardness : float or None
        Softplus hardness of the gamma clamp (None: hard clamp).
    ortho_kwargs : dict
        Keyword arguments for :func:`polar_decomposition`.
    """

    def __init__(
        self,
        equivectors,
        is_global=False,
        random=False,
        fix_params=False,
        mass_reg=None,
        gamma_max=None,
        gamma_hardness=None,
        ortho_kwargs=None,
    ):
        super().__init__()
        self.ortho_kwargs = {} if ortho_kwargs is None else dict(ortho_kwargs)
        self.equivectors = equivectors(n_vectors=3)
        self.is_global = is_global
        self.random = random
        self.mass_reg = mass_reg
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness
        if random or fix_params:
            self.equivectors.requires_grad_(False)

        # well-conditioned replacement vectors for masked particles: identity boost and
        # z/x rotation references; keeps the polar decomposition NaN-free everywhere
        self.register_buffer(
            "_pad_vecs",
            torch.tensor(
                [[1.0, 0, 0, 0], [0, 0, 0, 1.0], [0, 1.0, 0, 0]], dtype=torch.float32
            ),
            persistent=False,
        )

    def init_standardization(self, fourmomenta, mask):
        self.equivectors.init_standardization(fourmomenta, mask)

    def init_weights_or_not(self):
        if self.random and self.training:
            self.equivectors.apply(_init_weights)

    def mass_regularize(self, fourmomenta):
        if self.mass_reg is not None:
            mask = lorentz_squarednorm(fourmomenta) < self.mass_reg**2
            energy_reg = (fourmomenta[..., 1:] ** 2).sum(dim=-1).add(self.mass_reg**2).sqrt()
            energy = torch.where(mask, energy_reg, fourmomenta[..., 0])
            fourmomenta = torch.cat([energy.unsqueeze(-1), fourmomenta[..., 1:]], dim=-1)
        return fourmomenta

    def globalize_vecs_or_not(self, vecs, mask):
        if not self.is_global:
            return vecs
        m = mask[..., None, None].to(vecs.dtype)
        mean = (vecs * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return mean.expand_as(vecs)

    def forward(self, fourmomenta, scalars=None, mask=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta : torch.Tensor
            Four-momenta of shape (B, N, 4) in (E, px, py, pz).
        scalars : torch.Tensor or None
            Scalar features of shape (B, N, n_scalars).
        mask : torch.BoolTensor
            Valid-particle mask of shape (B, N).
        return_tracker : bool
            If True, additionally return a dict with regularization statistics.
        """
        if mask is None:
            mask = fourmomenta.new_ones(fourmomenta.shape[:-1], dtype=torch.bool)

        self.init_weights_or_not()
        fm = self.mass_regularize(fourmomenta)
        vecs = self.equivectors(fm, scalars=scalars, mask=mask)
        # replace masked particles' (zero) vectors with well-conditioned defaults so no
        # numerical regularization noise or NaN gradients can arise from padding
        vecs = torch.where(mask[..., None, None], vecs, self._pad_vecs.to(vecs.dtype))
        vecs = self.globalize_vecs_or_not(vecs, mask)
        boost = vecs[..., 0, :]
        rotation_references = vecs[..., 1:, :]
        boost, reg_gammamax, gamma_mean, gamma_max = clamp_boost(
            boost, gamma_max=self.gamma_max, gamma_hardness=self.gamma_hardness
        )

        trafo, reg_lightlike, reg_collinear = polar_decomposition(
            boost,
            rotation_references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        if not self.is_global:
            # identity frames for masked particles
            eye = torch.eye(4, device=trafo.device, dtype=trafo.dtype)
            trafo = torch.where(mask[..., None, None], trafo, eye)

        tracker = {
            "reg_lightlike": reg_lightlike,
            "reg_collinear": reg_collinear,
            "gamma_mean": gamma_mean,
            "gamma_max": gamma_max,
        }
        if reg_gammamax is not None:
            tracker["reg_gammamax"] = reg_gammamax
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


# ------------------------------------------------------------------------------------
# Weaver-facing jet tagger (ported from tagging-guide TransformerWrapper, dense path)
# ------------------------------------------------------------------------------------


class LLoCaTransformerTagger(nn.Module):
    """Weaver-facing LLoCa-Transformer jet tagger.

    Dense (zero-padded) port of the tagging-guide ``TransformerWrapper`` with a
    ``LearnedPDFrames`` frames-net: symmetry-breaking spurions are prepended and take part
    in the frames prediction (but are dropped afterwards), the four-momenta and the
    standard kinematic tagging features are expressed in each particle's local frame, and
    the per-jet logits are read off a global class token (or a masked mean when
    ``mean_aggregation=True``).

    The seven kinematic features (log pt, log E, log pt_rel, log E_rel, dphi, deta, dr)
    are computed internally, in the local frames; the weaver data config only needs to
    provide the extra (frame-independent) particle features via ``pf_features`` and the
    four-momenta via ``pf_vectors``. They can be reduced or switched off entirely via
    ``local_auxiliary_scalars``.

    Parameters
    ----------
    input_dim
        Number of extra scalar features per particle (``pf_features``).
    num_classes
        Number of output classes.
    attn_reps / num_heads / num_blocks / attention_factor / mlp_factor / dropout_prob /
    preserve_variance / elementwise_affine / checkpoint_blocks
        Forwarded to :class:`LLoCaTransformer` (defaults follow the tagging-guide
        ``lloca`` config at size 0).
    frames_is_global / frames_random / frames_fix_params / gamma_max / gamma_hardness /
    mass_reg
        Frames-net configuration (defaults follow the tagging-guide ``learnedpd`` config;
        ``mass_reg`` is in the units of the input four-momenta, GeV by default).
    equivectors_hidden_channels / equivectors_num_layers_mlp / equivectors_dropout_prob /
    equivectors_include_edges / equivectors_operation / equivectors_nonlinearity /
    equivectors_fm_norm / equivectors_layer_norm
        Equivariant vector-predictor configuration (defaults follow the tagging-guide
        ``equimlp`` config; ``hidden_channels`` follows the size-0 ``lloca`` config).
    ortho_method / ortho_eps_norm / ortho_eps_reg / ortho_eps_reg_lightlike /
    ortho_use_float64 / ortho_checks
        Orthogonalization configuration for the polar decomposition.
    beam_reference / two_beams / add_time_reference / spurion_scale
        Spurion configuration (defaults follow the tagging-guide ``tagging`` config).
    auxiliary_scalars
        Which (global-frame) kinematic features enter the frames-net scalars: 'all',
        'zinvariant', 'so3invariant', or None to disable them entirely (the frames-net
        then only sees the extra scalar features).
    local_auxiliary_scalars
        Which local-frame kinematic features enter the transformer: 'all' (the
        tagging-guide behaviour), 'zinvariant', 'so3invariant', or None to disable them
        entirely (the transformer then only sees the extra scalar features, and the
        local-frame four-momenta are never computed).
    mean_aggregation
        If True, aggregate with a masked mean over tokens instead of a class token.
    attention_backend
        ``"native"`` (default) runs the transformer on the dense zero-padded layout
        through ``torch.nn.functional.scaled_dot_product_attention``. ``"varlen"``
        (torch's native flash-attention varlen kernel, torch >= 2.10), ``"flash"``
        (the flash-attn package, FlashAttention-3 interface preferred), and
        ``"xformers"`` (``xformers.ops.memory_efficient_attention`` with a
        block-diagonal mask) drop the padding and run block-diagonal attention over the
        packed tokens instead (the frames-net stays dense). These packed backends
        require CUDA; on CPU the packed layout falls back to a materialized
        block-diagonal SDPA mask. ONNX export requires ``"native"``.
    momentum_float64
        Whether to run the frames-net and local-frame feature computation in float64
        (the tagging-guide default).
    trim
        Whether to enable sequence trimming during training.
    use_amp
        Whether to run the transformer under ``torch.autocast``.
    for_inference
        Whether to apply a softmax to the output (for deployment/ONNX export).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        # transformer configuration
        attn_reps: str = "8x0n+2x1n",
        num_heads: int = 8,
        num_blocks: int = 8,
        attention_factor: int = 1,
        mlp_factor: int = 2,
        dropout_prob: float | None = None,
        preserve_variance: bool = True,
        elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        # frames-net
        frames_is_global: bool = False,
        frames_random: bool = False,
        frames_fix_params: bool = False,
        gamma_max: float | None = None,
        gamma_hardness: float | None = 10.0,
        mass_reg: float | None = 5e-3,
        equivectors_hidden_channels: int = 32,
        equivectors_num_layers_mlp: int = 2,
        equivectors_dropout_prob: float | None = None,
        equivectors_include_edges: bool = True,
        equivectors_operation: str = "add",
        equivectors_nonlinearity: str = "softmax",
        equivectors_fm_norm: bool = True,
        equivectors_layer_norm: bool = True,
        ortho_method: str = "gramschmidt",
        ortho_eps_norm: float = 1e-15,
        ortho_eps_reg: float = 1e-16,
        ortho_eps_reg_lightlike: float = 1e-16,
        ortho_use_float64: bool = True,
        ortho_checks: bool = False,
        # spurions
        beam_reference: str | None = "all",
        two_beams: bool = True,
        add_time_reference: bool = True,
        spurion_scale: float = 1.0,
        # embedding / aggregation
        auxiliary_scalars: str | None = "all",
        local_auxiliary_scalars: str | None = "all",
        mean_aggregation: bool = False,
        momentum_float64: bool = True,
        # attention
        attention_backend: str = "native",
        # misc
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
        trim: bool = True,
        use_amp: bool = False,
        for_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        _logger.info("LLoCaTransformerTagger init-ed: %s", locals())

        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention_backend: {attention_backend}. "
                f"Supported backends: {ATTENTION_BACKENDS}."
            )
        self.mean_aggregation = mean_aggregation
        self.momentum_float64 = momentum_float64
        self.auxiliary_scalars = auxiliary_scalars
        self.local_auxiliary_scalars = local_auxiliary_scalars
        self.attention_backend = attention_backend
        self.use_amp = use_amp
        self.for_inference = for_inference

        spurions = get_spurion(beam_reference, add_time_reference, two_beams) * spurion_scale
        self.register_buffer("spurions", spurions, persistent=False)

        # frames-net; the equivectors see the extra scalars plus the (global-frame)
        # kinematic features
        num_scalars = input_dim + get_num_auxiliary_scalars(auxiliary_scalars)
        equivectors = partial(
            MLPVectors,
            num_scalars=num_scalars,
            hidden_channels=equivectors_hidden_channels,
            num_layers_mlp=equivectors_num_layers_mlp,
            dropout_prob=equivectors_dropout_prob,
            include_edges=equivectors_include_edges,
            operation=equivectors_operation,
            nonlinearity=equivectors_nonlinearity,
            fm_norm=equivectors_fm_norm,
            layer_norm=equivectors_layer_norm,
            use_amp=use_amp,
        )
        self.framesnet = LearnedPDFrames(
            equivectors,
            is_global=frames_is_global,
            random=frames_random,
            fix_params=frames_fix_params,
            mass_reg=mass_reg,
            gamma_max=gamma_max,
            gamma_hardness=gamma_hardness,
            ortho_kwargs=dict(
                use_float64=ortho_use_float64,
                method=ortho_method,
                eps_norm=ortho_eps_norm,
                eps_reg=ortho_eps_reg,
                eps_reg_lightlike=ortho_eps_reg_lightlike,
                checks=ortho_checks,
            ),
        )
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

        # the transformer sees the local-frame kinematic features, the extra scalars,
        # and (unless mean-aggregating) the global-token flag channel
        in_channels = (
            get_num_auxiliary_scalars(local_auxiliary_scalars)
            + input_dim
            + (0 if mean_aggregation else 1)
        )
        self.net = LLoCaTransformer(
            in_channels=in_channels,
            attn_reps=attn_reps,
            out_channels=num_classes,
            num_blocks=num_blocks,
            num_heads=num_heads,
            checkpoint_blocks=checkpoint_blocks,
            attention_factor=attention_factor,
            mlp_factor=mlp_factor,
            dropout_prob=dropout_prob,
            preserve_variance=preserve_variance,
            elementwise_affine=elementwise_affine,
            compile=compile_model,
            compile_kwargs=compile_kwargs,
        )

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

    def init_standardization(self, v, mask):
        """Initialize the frames-net edge standardization from a reference batch.

        Optional: if not called, the statistics are computed lazily from the first batch.

        Parameters
        ----------
        v : torch.Tensor
            Four-momenta of shape (N, 4, P) in (px, py, pz, energy).
        mask : torch.Tensor
            Mask of shape (N, 1, P) or (N, P).
        """
        if mask.dim() == 3:
            mask = mask.squeeze(1)
        mask = mask.bool()
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        if self.momentum_float64:
            fourmomenta = fourmomenta.to(torch.float64)
        fourmomenta = fourmomenta * mask.unsqueeze(-1)
        fourmomenta, mask, _ = self._prepend_spurions(fourmomenta, None, mask)
        self.framesnet.init_standardization(fourmomenta, mask)

    def _prepend_spurions(self, fourmomenta, scalars, mask):
        n_spurions = self.spurions.size(0)
        if n_spurions == 0:
            return fourmomenta, mask, scalars
        batch_size = fourmomenta.size(0)
        spurions = self.spurions.to(fourmomenta.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        fourmomenta = torch.cat([spurions, fourmomenta], dim=1)
        mask = torch.cat([mask.new_ones(batch_size, n_spurions), mask], dim=1)
        if scalars is not None:
            scalars = torch.cat(
                [scalars.new_zeros(batch_size, n_spurions, scalars.size(2)), scalars], dim=1
            )
        return fourmomenta, mask, scalars

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P) -- extra scalar features
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)
        mask = mask.squeeze(1).bool()  # (N, P)
        batch_size = x.size(0)

        scalars = x.transpose(1, 2)  # (N, P, C)
        # (E, px, py, pz) convention
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        if self.momentum_float64:
            fourmomenta = fourmomenta.to(torch.float64)
        # zero out padded entries (data configs may pad by wrapping real particles)
        fourmomenta = fourmomenta * mask.unsqueeze(-1)
        scalars = scalars * mask.unsqueeze(-1)

        # prepend spurions (zero scalar features, valid mask)
        n_spurions = self.spurions.size(0)
        fourmomenta, mask_spurions, scalars_spurions = self._prepend_spurions(
            fourmomenta, scalars, mask
        )

        jet = fourmomenta[:, n_spurions:].sum(dim=1, keepdim=True)  # (N, 1, 4)

        # global-frame kinematic features; zeroed on spurions and padding
        if self.auxiliary_scalars is None:
            framesnet_scalars = scalars_spurions
        else:
            aux = get_auxiliary_scalars(
                fourmomenta, jet, auxiliary_scalars=self.auxiliary_scalars
            )
            if n_spurions:
                aux = torch.cat(
                    [torch.zeros_like(aux[:, :n_spurions]), aux[:, n_spurions:]], dim=1
                )
            aux = aux * mask_spurions.unsqueeze(-1)
            aux = aux.to(scalars.dtype)
            framesnet_scalars = torch.cat([aux, scalars_spurions], dim=-1)

        # frames-net forward pass; spurions take part in the frames prediction
        frames_spurions, _tracker = self.framesnet(
            fourmomenta,
            scalars=framesnet_scalars,
            mask=mask_spurions,
            return_tracker=True,
        )

        # remove spurions
        frames = Frames(
            matrices=frames_spurions.matrices[:, n_spurions:],
            is_global=frames_spurions.is_global,
            det=frames_spurions.det[:, n_spurions:],
            inv=frames_spurions.inv[:, n_spurions:],
            is_identity=frames_spurions.is_identity,
        )
        fourmomenta = fourmomenta[:, n_spurions:]

        # transform features into the local frames
        if self.local_auxiliary_scalars is None:
            # local-frame kinematic features disabled: no need to transform anything
            features = scalars
        else:
            fourmomenta_local = self.trafo_fourmomenta(fourmomenta, frames)
            jet_local = self.trafo_fourmomenta(jet.expand_as(fourmomenta), frames)
            local_aux = get_auxiliary_scalars(
                fourmomenta_local, jet_local, auxiliary_scalars=self.local_auxiliary_scalars
            )
            # change dtype (see momentum_float64 option)
            features = torch.cat([local_aux.to(scalars.dtype), scalars], dim=-1)
        features = features * mask.unsqueeze(-1)
        frames.to(dtype=scalars.dtype)
        jet = jet.squeeze(1).to(scalars.dtype)  # (N, 4) reference momentum

        if self.attention_backend != "native":
            return self._forward_packed(features, frames, mask, jet)

        # handle global token: identity frame, one-hot flag in an extra scalar channel
        if not self.mean_aggregation:
            new_features = features.new_zeros(
                batch_size, features.size(1) + 1, features.size(2) + 1
            )
            new_features[:, 1:, :-1] = features
            new_features[:, 0, -1] = 1.0
            features = new_features
            mask = torch.cat([mask.new_ones(batch_size, 1), mask], dim=1)

            matrices_global = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
            )
            det_global = torch.ones(
                (batch_size, 1), device=frames.device, dtype=frames.dtype
            )
            frames = Frames(
                torch.cat([matrices_global, frames.matrices], dim=1),
                is_global=frames.is_global,
                det=torch.cat([det_global, frames.det], dim=1),
                inv=torch.cat([matrices_global, frames.inv], dim=1),
            )

        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, tokens)
        with torch.autocast(features.device.type, enabled=self.use_amp):
            outputs = self.net(
                inputs=features,
                frames=frames,
                p_ref=jet,
                attn_mask=attn_mask,
            )
        outputs = outputs * mask.unsqueeze(-1)

        # aggregation
        if self.mean_aggregation:
            output = outputs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
        else:
            output = outputs[:, 0]

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output

    def _forward_packed(self, features, frames, mask, jet):
        """Packed (sparse) forward path: drop the padding and run block-diagonal varlen
        attention over the concatenated tokens (port of the tagging-guide
        ``TransformerWrapper._forward_sparse``); the frames-net stays dense.

        Parameters
        ----------
        features : torch.Tensor
            Local-frame features of shape (B, P, C), zeroed on padding.
        frames : Frames
            Per-particle local frames of shape (B, P, 4, 4).
        mask : torch.BoolTensor
            Valid-particle mask of shape (B, P).
        jet : torch.Tensor
            Per-event reference (jet) momenta of shape (B, 4), energy-first.
        """
        # any upper bound on the per-event sequence lengths works; using the dense width
        # avoids a device-to-host sync
        maxlen = mask.size(1)
        [features, matrices, det, inv], batch, ptr = dense_to_sparse(
            [features, frames.matrices, frames.det, frames.inv], mask
        )

        if not self.mean_aggregation:
            # prepend a global token per event: identity frame, one-hot flag in an extra
            # scalar channel
            maxlen = maxlen + 1
            global_idxs, nonglobal_idxs, ptr, batch, num_total = insert_global_tokens(
                ptr, batch, features.shape[0]
            )
            new_features = features.new_zeros(num_total, features.shape[-1] + 1)
            new_features[nonglobal_idxs, :-1] = features
            new_features[:, -1].index_fill_(0, global_idxs, 1.0)
            features = new_features

            eye = torch.eye(4, device=matrices.device, dtype=matrices.dtype)
            matrices_new = eye.unsqueeze(0).expand(num_total, -1, -1).clone()
            matrices_new[nonglobal_idxs] = matrices
            inv_new = eye.unsqueeze(0).expand(num_total, -1, -1).clone()
            inv_new[nonglobal_idxs] = inv
            det_new = det.new_ones(num_total)
            det_new[nonglobal_idxs] = det
            matrices, det, inv = matrices_new, det_new, inv_new

        frames = Frames(
            matrices=matrices.unsqueeze(0),
            is_global=frames.is_global,
            det=det.unsqueeze(0),
            inv=inv.unsqueeze(0),
            is_identity=frames.is_identity,
        )
        # token-resolved reference momenta (global tokens use their event's jet)
        p_ref = jet.index_select(0, batch).unsqueeze(0)  # (1, tokens, 4)

        attn_kwargs = get_sparse_attention_kwargs(ptr, batch, maxlen, self.attention_backend)

        features = features.unsqueeze(0)  # (1, tokens, C)
        with torch.autocast(features.device.type, enabled=self.use_amp):
            outputs = self.net(
                inputs=features,
                frames=frames,
                p_ref=p_ref,
                **attn_kwargs,
            )
        outputs = outputs.squeeze(0)  # (tokens, num_classes)

        # aggregation
        if self.mean_aggregation:
            batch_size = ptr.numel() - 1
            counts = (ptr[1:] - ptr[:-1]).unsqueeze(-1).to(outputs.dtype)
            output = (
                outputs.new_zeros(batch_size, outputs.shape[-1]).index_add_(0, batch, outputs)
                / counts
            )
        else:
            output = outputs.index_select(0, global_idxs)

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output
