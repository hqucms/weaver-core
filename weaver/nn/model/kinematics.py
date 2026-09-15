"""Kinematic helper features for jet tagging.

The standardized tagging features (log pt, log E, log pt_rel, log E_rel, dphi, deta, dr)
and the four-momentum primitives they are built from, shared by the taggers in
:mod:`weaver.nn.model.LGATrSlim`, :mod:`weaver.nn.model.LLoCaTransformer` and
:mod:`weaver.nn.model.PlainTransformer`.

Ported from https://github.com/heidelberg-hepml/tagging-guide
(``experiments/hep.py`` and ``experiments/tagging/embedding.py``).

All functions take four-momenta of shape (..., 4) in the (E, px, py, pz) convention.
"""

from __future__ import annotations

import torch


_EPS_HEP = 1e-10

# weaver defaults for tagging features standardization (mean, factor); the features are
# standardized as (feature - mean) * factor
AUXILIARY_SCALARS_PREPROCESSING = [
    [1.7, 0.7],  # log_pt
    [2.0, 0.7],  # log_energy
    [-4.7, 0.7],  # log_pt_rel
    [-4.7, 0.7],  # log_energy_rel
    [0, 1],  # dphi
    [0, 1],  # deta
    [0.2, 4],  # dr
]


def stable_arctanh(x: torch.Tensor, eps: float = _EPS_HEP) -> torch.Tensor:
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def avoid_zero(x: torch.Tensor, eps: float = _EPS_HEP) -> torch.Tensor:
    # set small-abs values to eps for numerical stability
    return torch.where(x.abs() < eps, eps, x)


def get_pt(p: torch.Tensor) -> torch.Tensor:
    # transverse momentum of a (..., 4) tensor in (E, px, py, pz)
    return torch.sqrt((p[..., 1] ** 2 + p[..., 2] ** 2).clamp(min=_EPS_HEP))


def get_phi(p: torch.Tensor) -> torch.Tensor:
    # azimuthal angle
    return torch.arctan2(avoid_zero(p[..., 2]), avoid_zero(p[..., 1]))


def get_eta(p: torch.Tensor) -> torch.Tensor:
    # pseudo-rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1).clamp(min=_EPS_HEP))
    return stable_arctanh(p[..., 3] / p_abs)


def get_auxiliary_scalars(fourmomenta, jet, auxiliary_scalars="all", eps=1e-10):
    """Compute the standardized kinematic features typically used in jet tagging.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Particle four-momenta of shape (..., 4) in (E, px, py, pz).
    jet : torch.Tensor
        Jet four-momenta broadcastable against ``fourmomenta``.
    auxiliary_scalars : str or None
        Which features to include: 'all', 'zinvariant', 'so3invariant', or None.

    Returns
    -------
    torch.Tensor
        Features of shape (..., n_features); for 'all' these are
        (log_pt, log_energy, log_pt_rel, log_energy_rel, dphi, deta, dr).
    """
    log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
    log_energy = fourmomenta[..., 0].unsqueeze(-1).clamp(min=eps).log()

    log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
    log_energy_rel = (
        fourmomenta[..., 0].clamp(min=eps).log() - jet[..., 0].clamp(min=eps).log()
    ).unsqueeze(-1)
    phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
    dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
    deta = -(eta_4 - eta_jet).unsqueeze(-1)
    dr = torch.sqrt((dphi**2 + deta**2).clamp(min=eps))
    features = [
        log_pt,
        log_energy,
        log_pt_rel,
        log_energy_rel,
        dphi,
        deta,
        dr,
    ]
    for i, feature in enumerate(features):
        mean, factor = AUXILIARY_SCALARS_PREPROCESSING[i]
        features[i] = (feature - mean) * factor
    if auxiliary_scalars == "zinvariant":
        # exclude energy, because it is not invariant under z-boosts
        idx = [0, 2, 4, 5, 6]
    elif auxiliary_scalars == "so3invariant":
        # exclude everything except energy, because it is not invariant under SO(3) rotations
        idx = [1, 3]
    elif auxiliary_scalars is None:
        return torch.zeros(
            *features[0].shape[:-1], 0, device=fourmomenta.device, dtype=fourmomenta.dtype
        )
    elif auxiliary_scalars == "all":
        idx = list(range(len(features)))
    else:
        raise ValueError(f"auxiliary_scalars={auxiliary_scalars} not implemented")
    features = [features[i] for i in idx]
    return torch.cat(features, dim=-1)


def get_num_auxiliary_scalars(auxiliary_scalars="all") -> int:
    if auxiliary_scalars == "all":
        return 7
    elif auxiliary_scalars == "zinvariant":
        return 5
    elif auxiliary_scalars == "so3invariant":
        return 2
    elif auxiliary_scalars is None:
        return 0
    else:
        raise ValueError(f"auxiliary_scalars={auxiliary_scalars} not implemented")
