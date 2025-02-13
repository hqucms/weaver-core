from functools import lru_cache
from pathlib import Path

import torch

from ..utils.einsum import cached_einsum


@lru_cache()
def _load_geometric_product_tensor(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Loads geometric product tensor for geometric product between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.Device or str
        Device
    dtype : torch.Dtype
        Data type

    Returns
    -------
    basis : torch.Tensor with shape (16, 16, 16)
        Geometric product tensor
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        gmt = _load_geometric_product_tensor()
    else:
        filename = Path(__file__).parent.resolve() / "geometric_product.pt"
        gmt = torch.load(filename).to(torch.float32).to_dense()

    return gmt.to(device=device, dtype=dtype)


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product f(x,y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_geometric_product_tensor(device=x.device, dtype=x.dtype)

    # Compute geometric product
    outputs = cached_einsum("i j k, ... j, ... k -> ... i", gp, x, y)

    return outputs
