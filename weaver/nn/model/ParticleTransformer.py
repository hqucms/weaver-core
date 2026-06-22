"""Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""

import math

import copy
import numbers
from functools import partial
from typing import Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from weaver.utils.logger import _logger


def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    assert x.size(1) >= 2
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_p2(x, eps=1e-8):
    assert x.size(1) >= 3
    p2 = x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        p2 = p2.clamp(min=eps)
    return p2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - to_p2(x, eps=None)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def to_spherical(coords):
    # x: (N, 4, ...), dim1 : (x, y, z, t)
    x, y, z = coords[:, :3].split((1, 1, 1), dim=1)
    r = torch.sqrt(to_p2(coords, eps=None))
    theta = torch.acos((z / r.clamp(min=1e-8)).clamp(-1, 1))
    phi = torch.atan2(y, x)
    return r, theta, phi


def to_cylindrical(coords):
    # x: (N, 4, ...), dim1 : (x, y, z, t)
    x, y, z = coords[:, :3].split((1, 1, 1), dim=1)
    rho = torch.sqrt(to_pt2(coords, eps=None))
    phi = torch.atan2(y, x)
    return rho, phi, z


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def to_energy_momentum(x, return_unit_vector=True):
    energy = x[:, 3:4]
    mom = torch.sqrt(to_p2(x, eps=None))
    if return_unit_vector:
        return energy, mom, x[:, :3] / mom.clamp(min=1e-8)
    else:
        return energy, mom


def to_cos_sin_angles(xi, xj, normed_inputs=False, eps=1e-8):
    if normed_inputs:
        ni, nj = xi, xj
    else:
        ni, nj = p3_norm(xi, eps), p3_norm(xj, eps)
    cos = (ni * nj).sum(dim=1, keepdim=True).clamp(min=-1, max=1)
    sin = torch.linalg.cross(ni, nj, dim=1).norm(dim=1, keepdim=True).clamp(min=0, max=1)
    return cos, sin


def pairwise_lv_fts_pp(xi, xj, num_outputs=4, eps=1e-8):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def pairwise_lv_fts_ee(xi, xj, num_outputs=6, eps=1e-8):
    # outputs: [lnm2, cos_angle, sin_angle, lnkt, lnz, lnjade]
    lnm2 = torch.log(to_m2(xi + xj, eps=eps))
    outputs = [lnm2]

    if num_outputs > 1:
        ei, pi, ni = to_energy_momentum(xi)
        ej, pj, nj = to_energy_momentum(xj)
        cos_angle, sin_angle = to_cos_sin_angles(ni, nj, normed_inputs=True)
        outputs += [cos_angle, sin_angle]

    if num_outputs > 3:
        pmin = torch.minimum(pi, pj)
        lnkt = torch.log((pmin * sin_angle).clamp(min=eps))
        lnz = torch.log((pmin / (pi + pj).clamp(min=eps)).clamp(min=eps))
        outputs += [lnkt, lnz]

    if num_outputs > 5:
        lnjade = torch.log((ei * ej * (1 - cos_angle)).clamp(min=eps))
        outputs.append(lnjade)

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def pairwise_lv_fts_xyzt(xi, xj, num_outputs=7, coords="rectangular", eps=1e-8):
    # outputs: [ln_dist2, cos_angle, sin_angle, ln_dt] + coords-diff
    dij = xi - xj
    ln_dist2 = torch.log(to_p2(dij, eps=eps))
    outputs = [ln_dist2]

    if num_outputs > 1:
        ei, pi, ni = to_energy_momentum(xi)
        ej, pj, nj = to_energy_momentum(xj)
        cos_angle, sin_angle = to_cos_sin_angles(ni, nj, normed_inputs=True)
        outputs += [cos_angle, sin_angle]

    if num_outputs > 3:
        ln_dt = torch.asinh(dij[:, 3:4])
        outputs += [ln_dt]

    if num_outputs > 4:
        if coords.lower().startswith("rect") or coords.lower().startswith("cart"):
            # rectangular/Cartesian coordinate system
            dx, dy, dz = dij[:, :3].split((1, 1, 1), dim=1)
            outputs += [dx, dy, dz]
        elif coords.lower().startswith("sph"):
            # spherical coordinate system
            r_i, theta_i, phi_i = to_spherical(xi)
            r_j, theta_j, phi_j = to_spherical(xj)
            outputs += [torch.asinh(r_i - r_j), theta_i - theta_j, delta_phi(phi_i, phi_j)]
        elif coords.lower().startswith("cyl"):
            # cylindrical coordinate system
            rho_i, phi_i, z_i = to_cylindrical(xi)
            rho_j, phi_j, z_j = to_cylindrical(xj)
            outputs += [torch.asinh(rho_i - rho_j), delta_phi(phi_i, phi_j), torch.asinh(z_i - z_j)]
        else:
            raise RuntimeError(f"Unrecognized coordinate system name {coords}")

    assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat(
        (
            torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
            torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
            idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
            idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        i, uu.flatten(), size=(batch_size, num_fts, seq_len + 1, seq_len + 1), device=uu.device
    ).to_dense()[:, :, :seq_len, :seq_len]


def tril_indices(row, col, offset=0, *, dtype=torch.long, device="cpu"):
    return torch.ones(row, col, dtype=dtype, device=device).tril(offset).nonzero().T


class SequenceTrimmer(nn.Module):
    def __init__(
        self, enabled=False, target=(0.9, 1.02), warmup_steps=5, round_to_32=False, num_extra_tokens=0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self.warmup_steps = warmup_steps
        self.round_to_32 = round_to_32
        self.num_extra_tokens = num_extra_tokens
        self.register_buffer("_counter", torch.LongTensor([0]), persistent=False)

    @torch._dynamo.disable
    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < self.warmup_steps:
                self._counter.add_(1)
            else:
                seq_len = mask.size(-1)
                if v is not None:
                    if not isinstance(v, (list, tuple)):
                        v = [v]
                if self.training:
                    # Use torch RNG instead of Python random to avoid graph breaks
                    q = torch.empty(1, device=mask.device).uniform_(*self.target).clamp_(max=1)
                    maxlen = torch.quantile(mask.float().sum(dim=-1), q).long()
                    if self.round_to_32:
                        # Round up to next multiple of 32
                        # effectively: ceil(x / 32) * 32
                        target_len = maxlen + self.num_extra_tokens
                        target_len = ((target_len + 31) // 32) * 32
                        target_len = torch.clamp(target_len, min=32)
                        maxlen = torch.clamp(target_len, max=seq_len) - self.num_extra_tokens
                    rand = torch.rand_like(mask.float())
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = [torch.gather(_v, -1, perm.expand_as(_v)) for _v in v]
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = maxlen.clamp(min=1)
                if maxlen < seq_len:
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = [_v[:, :, :maxlen] for _v in v]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]
                if v is not None:
                    if len(v) == 1:
                        v = v[0]

        return x, v, mask, uu


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization, exportable to ONNX.

    This is an ONNX-export-only stand-in for ``torch.nn.RMSNorm``. The native
    ``nn.RMSNorm`` lowers to ``aten::rms_norm``, for which the TorchScript-based
    ONNX exporter has no symbolic (it fails for every opset), so models using it
    (ParticleTransformer ``version >= 3``) cannot be exported. This manual
    implementation decomposes into primitive ops that export cleanly.

    It mirrors ``nn.RMSNorm`` numerically (same ``eps`` default of the input
    dtype's epsilon) and is state-dict compatible (single ``weight`` parameter),
    so a model trained with ``nn.RMSNorm`` can be loaded into one built with this
    class for export. ``nn.RMSNorm`` is kept for training (fused kernel), see
    ``ParticleTransformer``/``Block`` which select the implementation based on
    ``for_inference``.
    """

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # dims over which to compute the RMS (the trailing `normalized_shape` dims)
        self._dims = tuple(range(-len(self.normalized_shape), 0))
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        variance = x.pow(2).mean(dim=self._dims, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        if self.weight is not None:
            x_normed = x_normed * self.weight
        return x_normed

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu", use_conv_embed=False):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.use_conv_embed = bool(use_conv_embed)
        if self.use_conv_embed:
            assert normalize_input == True
            module_list = []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=False),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            self.conv_embed = nn.Sequential(*module_list[:-1])
            self.embed = nn.Identity()
        else:
            module_list = []
            for dim in dims:
                module_list.extend(
                    [
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            self.conv_embed = nn.Identity()
            self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = self.conv_embed(x)
            x = x.transpose(1, 2).contiguous()
        # x: (batch, seq_len, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        pairwise_lv_type="pp",
        remove_self_pair=False,
        use_pre_activation_pair=True,
        normalize_input=True,
        activation="gelu",
        use_bias=True,
        eps=1e-8,
        for_onnx=False,
        sparse_eval=(True, True),
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.remove_self_pair = remove_self_pair
        self.for_onnx = for_onnx
        self.sparse_eval = (False, False) if for_onnx else sparse_eval  # flags are for (train, inference)
        self.tril_indices_fn = tril_indices if self.for_onnx else torch.tril_indices
        self.out_dim = dims[-1]

        if pairwise_lv_type == "pp":
            self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(pairwise_lv_fts_pp, num_outputs=pairwise_lv_dim, eps=eps)
        elif pairwise_lv_type == "ee":
            self.is_symmetric = (pairwise_lv_dim <= 6) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(pairwise_lv_fts_ee, num_outputs=pairwise_lv_dim, eps=eps)
        elif pairwise_lv_type.startswith("xyzt"):
            coords = pairwise_lv_type.split(":")[1] if ":" in pairwise_lv_type else None
            self.is_symmetric = (pairwise_lv_dim <= 3) and (pairwise_input_dim == 0)
            self.pairwise_lv_fts = partial(pairwise_lv_fts_xyzt, num_outputs=pairwise_lv_dim, coords=coords, eps=eps)
        else:
            raise RuntimeError("Invalid value for `pairwise_lv_type`: " + pairwise_lv_type)

        if pairwise_lv_dim > 0:
            input_dim = pairwise_lv_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=use_bias),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)

        if pairwise_input_dim > 0:
            input_dim = pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1, bias=use_bias),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.fts_embed = nn.Sequential(*module_list)

    def _embed_pairs(self, x, uu):
        """Run embedding networks on pair features. Returns (batch_or_1, out_dim, num_pairs)."""
        elements = None
        if x is not None:
            elements = self.embed(x)
        if uu is not None:
            fts = self.fts_embed(uu)
            elements = fts if elements is None else elements + fts
        return elements

    def _forward_dense(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        if x is not None:
            batch_size, _, seq_len = x.size()
        else:
            batch_size, _, seq_len, _ = uu.size()

        if self.is_symmetric:
            i, j = self.tril_indices_fn(
                seq_len,
                seq_len,
                offset=-1 if self.remove_self_pair else 0,
                device=(x if x is not None else uu).device,
            )
            if x is not None:
                xi = x[:, :, i]  # (batch, dim, num_tril_pairs)
                xj = x[:, :, j]
                x = self.pairwise_lv_fts(xi, xj)
            if uu is not None:
                # (batch, dim, num_tril_pairs)
                uu = uu[:, :, i, j]
        else:
            if x is not None:
                x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                if self.remove_self_pair:
                    diag_idx = torch.arange(0, seq_len, device=x.device)
                    x[:, :, diag_idx, diag_idx] = 0
                x = x.reshape(batch_size, self.pairwise_lv_dim, seq_len * seq_len)
            if uu is not None:
                uu = uu.reshape(batch_size, self.pairwise_input_dim, seq_len * seq_len)

        elements = self._embed_pairs(x, uu)

        if self.is_symmetric:
            y = elements.new_zeros(batch_size, self.out_dim, seq_len, seq_len)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.reshape(batch_size, self.out_dim, seq_len, seq_len)
        return y

    def _forward_sparse(self, x, uu=None, mask=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        if x is not None:
            batch_size, _, seq_len = x.size()
        else:
            batch_size, _, seq_len, _ = uu.size()

        i0, i2, i3 = (Ellipsis,) * 3
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (batch_size, 1, seq_len, seq_len)
            if self.is_symmetric:
                offset = -1 if self.remove_self_pair else 0
                i0, _, i2, i3 = pair_mask.float().tril(offset).nonzero(as_tuple=True)
            else:
                i0, _, i2, i3 = pair_mask.nonzero(as_tuple=True)

        if x is not None:
            x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
            x = x.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_lv_dim)
            x = x.T.unsqueeze(0)  # (1, pairwise_lv_dim, num_elements)
        if uu is not None:
            uu = uu.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, pairwise_input_dim)
            uu = uu.T.unsqueeze(0)  # (1, pairwise_input_dim, num_elements)

        elements = self._embed_pairs(x, uu)
        elements = elements.squeeze(0).T  # (num_elements, out_dim)

        y = torch.zeros(batch_size, seq_len, seq_len, self.out_dim, dtype=elements.dtype, device=elements.device)
        y[i0, i2, i3, :] = elements
        if self.is_symmetric:
            y[i0, i3, i2, :] = elements
        y = y.permute(0, 3, 1, 2).contiguous()

        return y

    def forward(self, x, uu=None, mask=None):
        sparse_eval = self.sparse_eval[0 if self.training else 1]
        if sparse_eval and mask is not None:
            return self._forward_sparse(x, uu=uu, mask=mask)
        else:
            return self._forward_dense(x, uu=uu, mask=mask)


def _canonical_mask(
    mask: Optional[torch.Tensor],
    mask_name: str,
    other_type: Optional[Any],
    other_name: str,
    target_type: Any,
    check_other: bool = True,
) -> Optional[torch.Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


def _none_or_dtype(input: Optional[torch.Tensor]):
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


class Attention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        use_qk_norm=False,
        norm_layer=nn.LayerNorm,
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if use_qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        assert not (headwise_attn_output_gate and elementwise_attn_output_gate)
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        if self.headwise_attn_output_gate:
            self.gate_proj = torch.nn.Linear(embed_dim, num_heads, bias=bias, **factory_kwargs)
        elif self.elementwise_attn_output_gate:
            self.gate_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.use_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
            self.use_sdpa = False

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):

        for k in list(state_dict.keys()):
            if k.endswith("in_proj_weight"):
                state_dict[k.replace("_weight", ".weight")] = state_dict.pop(k)
            elif k.endswith("in_proj_bias"):
                state_dict[k.replace("_bias", ".bias")] = state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        # (bsz, src_len)
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        # (bsz, num_heads, tgt_len, src_len)
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), (
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            )
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                assert attn_mask.shape == (bsz, self.num_heads, tgt_len, src_len), (
                    f"expecting attn_mask shape of {(bsz, self.num_heads, tgt_len, src_len)}, but got {attn_mask.shape}"
                )
                attn_mask = attn_mask + key_padding_mask

        # (bsz, seq_len, num_heads*head_dim)
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)

        # -> (bsz, num_heads, src/tgt_len, head_dim)
        q = self.q_norm(q.view(bsz, tgt_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(bsz, src_len, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        if self.use_sdpa:
            # attn_output: (bsz, num_heads, tgt_len, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        else:
            q_scaled = q * math.sqrt(1.0 / float(self.head_dim))  # (bsz, num_heads, tgt_len, head_dim)
            attn_weight = q_scaled @ k.transpose(-2, -1)  # (bsz, num_heads, tgt_len, src_len)
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = F.softmax(attn_weight, dim=-1)
            if dropout_p > 0:
                attn_weight = F.dropout(attn_weight, p=dropout_p)
            attn_output = attn_weight @ v  # (bsz, num_heads, tgt_len, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, tgt_len, num_heads, head_dim)

        if self.headwise_attn_output_gate:
            gate_score = torch.sigmoid(self.gate_proj(query))  # (bsz, tgt_len, num_heads)
            attn_output = attn_output * gate_score.reshape(bsz, tgt_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            gate_score = torch.sigmoid(self.gate_proj(query))  # (bsz, tgt_len, num_heads * head_dim)
            attn_output = attn_output * gate_score.reshape(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        activation="gelu",
        use_rmsnorm=False,
        use_bias=True,
        use_qk_norm=False,
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        layer_scale_init_values=None,
        drop_path_rate=0.0,
        scale_attn_mask=False,
        scale_attn=True,
        scale_fc=True,
        scale_heads=True,
        scale_resids=True,
        for_inference=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        # use the ONNX-exportable `RMSNorm` only for inference/export; keep the
        # native (fused) `nn.RMSNorm` for training performance
        norm_layer = (RMSNorm if for_inference else nn.RMSNorm) if use_rmsnorm else nn.LayerNorm
        self.pre_attn_norm = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            bias=use_bias,
            use_qk_norm=use_qk_norm,
            norm_layer=norm_layer,
            headwise_attn_output_gate=headwise_attn_output_gate,
            elementwise_attn_output_gate=elementwise_attn_output_gate,
        )
        self.post_attn_norm = norm_layer(embed_dim) if scale_attn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.ls1 = (
            LayerScale(embed_dim, init_values=layer_scale_init_values) if layer_scale_init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.pre_fc_norm = norm_layer(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim, bias=use_bias)
        if activation == "swiglu":
            self.fc1_g = nn.Linear(embed_dim, self.ffn_dim, bias=use_bias)
            self.act = nn.SiLU()
        else:
            self.fc1_g = None
            self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = norm_layer(self.ffn_dim) if scale_fc else nn.Identity()
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim, bias=use_bias)
        self.ls2 = (
            LayerScale(embed_dim, init_values=layer_scale_init_values) if layer_scale_init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.c_mask = nn.Parameter(torch.ones(1), requires_grad=True) if scale_attn_mask else None
        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, seq_len, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(batch, 1, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``True``.

        Returns:
            encoded output of shape `(batch, seq_len, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=1)  # (batch, 1+seq_len, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (batch, 1, embed_dim)
        else:
            if self.c_mask is not None and attn_mask is not None:
                attn_mask = torch.mul(self.c_mask, attn_mask)
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]  # (batch, seq_len, embed_dim)

        if self.c_attn is not None:
            bsz, tgt_len, _ = x.size()
            x = x.view(bsz, tgt_len, self.num_heads, self.head_dim)
            x = x * self.c_attn.view(1, 1, self.num_heads, 1)
            x = x.reshape(bsz, tgt_len, self.embed_dim)
        x = self.post_attn_norm(x)
        x = self.dropout(x)
        x = self.drop_path1(self.ls1(x))
        x = x + residual

        residual = x
        x = self.pre_fc_norm(x)
        if self.fc1_g is None:
            x = self.act(self.fc1(x))
        else:
            x_gate = self.fc1_g(x)
            x = self.fc1(x)
            x = self.act(x_gate) * x
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.drop_path2(self.ls2(x))
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = x + residual

        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        pair_input_type="pp",
        pair_input_dim=None,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        use_conv_embed=False,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        pair_embed_sparse_eval=None,
        num_heads=8,
        num_layers=8,
        block_params=None,
        block_ids_with_attn_mask=None,
        include_global_token=False,
        num_cls_layers=2,
        cls_block_params=None,
        fc_params=(),
        activation="gelu",
        # misc
        version=1,
        weight_init="moco",
        fix_init=True,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        use_amp=False,
        compile_model=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        _logger.info("ParticleTransformer init-ed: %s", locals())

        self.for_inference = for_inference
        self.for_segmentation = for_segmentation
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            activation=activation,
            use_rmsnorm=False,
            use_bias=True,
            use_qk_norm=False,
            headwise_attn_output_gate=False,
            elementwise_attn_output_gate=False,
            layer_scale_init_values=None,
            drop_path_rate=0.0,
            scale_attn_mask=False,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
            for_inference=for_inference,
        )
        if version > 1:
            default_cfg.update(
                activation="swiglu",
                scale_fc=False,
                scale_attn=False,
                scale_heads=False,
                scale_resids=False,
            )
        if version > 2:
            default_cfg.update(
                use_rmsnorm=True,
                use_bias=False,
                dropout=0,
                attn_dropout=0,
                activation_dropout=0,
                drop_path_rate=0.1,
            )
            # TODO: still experimental
            if version == 3.5:
                default_cfg.update(
                    use_qk_norm=True,
                    elementwise_attn_output_gate=True,
                )
            if version == 3.6:
                default_cfg.update(
                    use_qk_norm=True,
                    elementwise_attn_output_gate=True,
                )
                include_global_token = True
                num_cls_layers = 0
        # use the ONNX-exportable `RMSNorm` only for inference/export; keep the
        # native (fused) `nn.RMSNorm` for training performance
        norm_layer = (RMSNorm if for_inference else nn.RMSNorm) if default_cfg["use_rmsnorm"] else nn.LayerNorm

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info("cfg_block: %s" % str(cfg_block))

        if num_cls_layers > 0:
            cfg_cls_block = copy.deepcopy(default_cfg)
            cfg_cls_block.update({"dropout": 0, "attn_dropout": 0, "activation_dropout": 0})
            if cls_block_params is not None:
                cfg_cls_block.update(cls_block_params)
            _logger.info("cfg_cls_block: %s" % str(cfg_cls_block))

        if block_ids_with_attn_mask is None:
            self.block_ids_with_attn_mask = [True] * num_layers
        else:
            self.block_ids_with_attn_mask = [False] * num_layers
            for idx in block_ids_with_attn_mask:
                self.block_ids_with_attn_mask[idx] = True
        _logger.info("block w/ attn_mask: %s" % str(self.block_ids_with_attn_mask))

        self.embed = (
            Embed(input_dim, embed_dims, activation=activation, use_conv_embed=use_conv_embed)
            if len(embed_dims) > 0
            else nn.Identity()
        )

        if pair_input_dim is None:
            if pair_input_type == "pp":
                pair_input_dim = 4
            elif pair_input_type == "ee":
                pair_input_dim = 6
            elif pair_input_type.startswith("xyzt"):
                pair_input_dim = 7
        self.pair_extra_dim = pair_extra_dim
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                pair_extra_dim,
                (*pair_embed_dims, cfg_block["num_heads"]),
                pairwise_lv_type=pair_input_type,
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                use_bias=default_cfg["use_bias"],
                for_onnx=for_inference,
                sparse_eval=((False, True) if compile_model else (True, True))
                if pair_embed_sparse_eval is None else pair_embed_sparse_eval,
            )
            if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
            else None
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = (
            nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)]) if num_cls_layers > 0 else None
        )
        self.norm = norm_layer(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for param in fc_params:
                try:
                    out_dim, drop_rate, act = param
                except ValueError:
                    (out_dim, drop_rate), act = param, "relu"
                if act == "swiglu":
                    layer = nn.Sequential(
                        SwiGLUFFN(in_dim, out_dim * 4, out_dim, drop=drop_rate, bias=default_cfg["use_bias"]),
                        norm_layer(out_dim),
                    )
                else:
                    layer = nn.Sequential(
                        nn.Linear(in_dim, out_dim, bias=default_cfg["use_bias"]),
                        nn.GELU() if act == "gelu" else nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                fcs.append(layer)
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        assert not (include_global_token and num_cls_layers > 0)
        self.include_global_token = include_global_token

        # cls tokens
        if self.include_global_token or (not self.for_segmentation and num_cls_layers > 0):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # sequence trimmer
        num_extra_tokens = 1 if self.include_global_token else 0
        self.trimmer = SequenceTrimmer(
            enabled=trim and not for_inference, round_to_32=compile_model, num_extra_tokens=num_extra_tokens
        )

        # weight initialization
        if weight_init is not None:
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("timm", "moco")
        if mode == "timm":
            named_apply(init_weights_vit_timm, self)
        elif mode == "moco":
            named_apply(init_weights_vit_moco, self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
        with torch.set_grad_enabled(x.requires_grad):
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (batch_size, seq_len) -- `True`` means masked/ignored

        # input embedding
        x = self.embed(x).masked_fill(~mask.transpose(1, 2), 0)  # (batch_size, seq_len, embed_dim)
        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu=uu, mask=mask)  # (batch_size, num_heads, seq_len, seq_len)

        if self.include_global_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (batch, 1+seq_len, embed_dim)
            padding_mask = F.pad(padding_mask, (1, 0), mode="constant", value=False)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (1, 0, 1, 0), mode="constant", value=0)

        # transform
        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attn_mask if self.block_ids_with_attn_mask[idx] else None,
            )

        # x: (batch, seq_len, embed_dim)
        # padding_mask: (batch, seq_len)
        return x, padding_mask

    def _forward_aggregator(self, x, padding_mask):
        if self.include_global_token:
            cls_tokens = x[:, 0, :]  # (batch, embed_dim)
        elif self.cls_blocks is not None:
            # for classification: extract using class token
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch, 1, embed_dim)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)  # (batch, 1, embed_dim)
            cls_tokens = cls_tokens.squeeze(1)  # (batch, embed_dim)
        else:
            # for classification: simple average pooling
            mask = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            counts = mask.float().sum(dim=1).clamp(min=1)  # (batch, 1)
            cls_tokens = (x * mask).sum(dim=1) / counts  # (batch, embed_dim)

        x_cls = self.norm(cls_tokens)  # (batch, embed_dim)
        return x_cls

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (batch_size, num_fts, seq_len)
        # v: (batch_size, 4, seq_len) [px,py,pz,energy]
        # mask: (batch_size, 1, seq_len) -- real particle = 1, padded = 0
        # for pytorch: uu (batch_size, C', num_pairs), uu_idx (batch_size, 2, num_pairs)
        # for onnx: uu (batch_size, C', seq_len, seq_len), uu_idx=None

        x, padding_mask = self._forward_encoder(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx)

        if self.cls_blocks is None and self.fc is None:
            # x: (batch, seq_len, embed_dim)
            # padding_mask: (batch, seq_len)
            return x, padding_mask

        # === for segmentation ===
        if self.for_segmentation:
            if self.include_global_token:
                x = x[:, 1:, :]
            x = self.norm(x)
            if self.fc is not None:
                x = self.fc(x)
            # x: (batch, seq_len, embed_dim) -> output: (batch, embed_dim, seq_len)
            output = x.transpose(1, 2).contiguous()
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output

        x_cls = self._forward_aggregator(x, padding_mask)
        if self.fc is None:
            return x_cls

        # fc
        output = self.fc(x_cls)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class ParticleTransformerTagger(nn.Module):
    def __init__(
        self,
        pf_input_dim,
        sv_input_dim,
        num_classes=None,
        # network configurations
        pair_input_type="pp",
        pair_input_dim=None,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params=None,
        fc_params=(),
        activation="gelu",
        # misc
        version=1,
        weight_init="moco",
        fix_init=True,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        use_amp=False,
        compile_model=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(
            input_dim=embed_dims[-1],
            num_classes=num_classes,
            # network configurations
            pair_input_type=pair_input_type,
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=[],
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=fc_params,
            activation=activation,
            # misc
            version=version,
            weight_init=weight_init,
            fix_init=fix_init,
            trim=False,
            for_inference=for_inference,
            for_segmentation=for_segmentation,
            use_amp=use_amp,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "part.cls_token",
        }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        pf_x = self.pf_embed(pf_x)  # after embed: (batch, seq_len, embed_dim)
        sv_x = self.sv_embed(sv_x)
        x = torch.cat([pf_x, sv_x], dim=1)

        return self.part(x, v, mask)


class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):
    def __init__(
        self,
        pf_input_dim,
        sv_input_dim,
        num_classes=None,
        # network configurations
        pair_input_type="pp",
        pair_input_dim=None,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params=None,
        fc_params=(),
        activation="gelu",
        # misc
        version=1,
        weight_init="moco",
        fix_init=True,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        use_amp=False,
        compile_model=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.for_inference = for_inference

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(
            input_dim=embed_dims[-1],
            num_classes=num_classes,
            # network configurations
            pair_input_type=pair_input_type,
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=[],
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=fc_params,
            activation=activation,
            # misc
            version=version,
            weight_init=weight_init,
            fix_init=fix_init,
            trim=False,
            for_inference=for_inference,
            for_segmentation=for_segmentation,
            use_amp=use_amp,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "part.cls_token",
        }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            if not self.for_inference:
                if pf_uu_idx is not None:
                    pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))

            pf_x, pf_v, pf_mask, pf_uu = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)
            uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
            uu[:, :, : pf_x.size(2), : pf_x.size(2)] = pf_uu

        pf_x = self.pf_embed(pf_x)  # after embed: (batch, seq_len, embed_dim)
        sv_x = self.sv_embed(sv_x)
        x = torch.cat([pf_x, sv_x], dim=1)

        return self.part(x, v, mask, uu)


### weight initialization methods ###
def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "in_proj" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
