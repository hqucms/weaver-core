from functools import lru_cache

import torch
import clifford
import numpy as np

from gatr.utils.einsum import cached_einsum, custom_einsum

# switch to decide whether to use the full Lorentz group ('False')
# or the special orthochronous Lorentz group ('True')
# They only differ in the construction of linear maps in _compute_pin_equi_linear_basis
USE_FULLY_CONNECTED_SUBGROUP = True


@lru_cache()
def _compute_pin_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32, normalize=True
) -> torch.Tensor:
    """Constructs basis elements for Pin(1,3)-equivariant linear maps between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype
    normalize : bool
        Whether to normalize the basis elements

    Returns
    -------
    basis : torch.Tensor with shape (NUM_PIN_LINEAR_BASIS_ELEMENTS, 16, 16)
        Basis elements for equivariant linear maps.
    """

    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        basis = _compute_pin_equi_linear_basis(normalize=normalize)
    else:
        # explicit construction of a pin-equilinear basis for the Lorentz group
        layout, blades = clifford.Cl(1, 3)
        linear_basis = []
        mults = [1, layout.pseudoScalar] if USE_FULLY_CONNECTED_SUBGROUP else [1]
        for mult in mults:
            for grade in range(5):
                w = np.stack([(x(grade) * mult).value for x in blades.values()], 1)
                w = w.astype(np.float32)
                if normalize:
                    # w /= np.linalg.norm(w) # straight-forward normalization
                    w /= np.linalg.svd(w)[1].max()  # alternative normalization
                linear_basis.append(w)
        linear_basis = np.stack(linear_basis)

        basis = torch.tensor(linear_basis, dtype=torch.float32).to_dense()
    return basis.to(device=device, dtype=dtype)


@lru_cache()
def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor with shape (16,)
        The diagonal of the reversal matrix, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache()
def _compute_grade_involution(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Constructs a matrix that computes multivector grade involution.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    involution_diag : torch.Tensor with shape (16,)
        The diagonal of the involution matrix, consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map f(x) = sum_{a,j} coeffs_a W^a_ij x_j.

    The W^a are seven pre-defined basis elements.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between x and coeffs.
    coeffs : torch.Tensor with shape (out_channels, in_channels, 10)
        Coefficients for the basis elements. Batch dimensions must be broadcastable between x and
        coeffs.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x and coeffs.
    """
    basis = _compute_pin_equi_linear_basis(device=x.device, dtype=x.dtype)
    return custom_einsum(
        "y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1]
    )


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input tensor to the individual grades.

    The return value is a single tensor with a new grade dimension.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5, 16)
        Output multivector. The second-to-last dimension indexes the grades.
    """

    # Select kernel on correct device
    basis = _compute_pin_equi_linear_basis(
        device=x.device, dtype=x.dtype, normalize=False
    )

    # First five basis elements are grade projections
    basis = basis[:5]

    # Project to grades
    projections = cached_einsum("g i j, ... j -> ... g i", basis, x)

    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    The reversal has the same scalar, vector, and pseudoscalar components, but flips sign in the
    bivector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a multivector.

    The reversal has the same scalar, bivector, and pseudoscalar components, but flips sign in the
    vector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
