import torch


def embed_vector(vector: torch.Tensor) -> torch.Tensor:
    """Embeds Lorentz vectors in multivectors.

    Parameters
    ----------
    vector : torch.Tensor with shape (..., 4)
        Lorentz vector

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = vector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=vector.dtype, device=vector.device
    )

    # Embedding into Lorentz vectors
    multivector[..., 1:5] = vector

    return multivector


def extract_vector(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector, extract a Lorentz vector.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.

    Returns
    -------
    vector : torch.Tensor with shape (..., 4)
        Lorentz vector
    """

    vector = multivector[..., 1:5]

    return vector
