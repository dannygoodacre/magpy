"""This file provivdes some useful matrix operations.

References
----------

.. [1] Baldwin, A. J. and Jones, J. A., “Efficiently computing the Uhlmann 
       fidelity for density matrices”, *Physical Review A*, vol. 107, 
       no. 1, Art. no. 012427, 2023. doi:10.1103/PhysRevA.107.012427.

"""


import functools

import torch
from torch import Tensor


def frob(a: Tensor, b: Tensor) -> Tensor:
    """The Frobenius inner product.

    If `a` is a 3D tensor and `b` is a 2D tensor, then the inner product is
    batched across `a`.

    If both are 3D tensors, then the inner product is applied between each
    2D tensor within.

    Parameters
    ----------
    a : Tensor
        A 2D tensor or 3D tensor
    b : Tensor
        A 2D tensor or 3D tensor

    Returns
    -------
    Tensor
        Resultant (batch) inner product
    """

    try:
        return torch.sum(torch.conj(a) * b, dim=(1, 2))

    except IndexError:
        return torch.sum(torch.conj(a) * b)
    

# TODO: Profile this against other approaches: balanced tree, reshape, etc.
def kron(*args: Tensor) -> Tensor:
    """The Kronecker product."""

    return functools.reduce(torch.kron, args)


def msqrt_herm(a: Tensor) -> Tensor:
    """The square root of a Hermitian matrix.

    If `a` is a 3D tensor then the operation is applied to each
    2D tensor within.

    Parameters
    ----------
    a : Tensor
        A 2D tensor or a 3D tensor
        
    Returns
    -------
    Tensor
        Resultant (batch) matrix square root
    """
    
    eigvals, eigvecs = torch.linalg.eigh(a)
    
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))

    return eigvecs @ torch.diag_embed(sqrt_eigvals.to(a.dtype)) @ torch.conj(eigvecs.transpose(-2, -1))


def timegrid(start: float, stop: float, step: float) -> Tensor:
    """A grid of values across the given interval with the specified spacing.

    Parameters
    ----------
    start : float
        Start of interval
    stop : float
        End of interval
    step : float
        Spacing of points in interval

    Returns
    -------
    Tensor
        Grid of values
    """

    return torch.arange(start, stop + step, step)


def uhlmann(a: Tensor, b: Tensor) -> Tensor:
    """The Uhlmann fidelity.
    
    If `a` is a 3D tensor then the operation is applied to each
    2D tensor within.

    Parameters
    ----------
    a : Tensor
        A 2D tensor or a 3D tensor Hermitian, PSD matrices
    b : Tensor
        A 2D tensor or a 3D tensor of Hermitian, PSD matrices

    Returns
    -------
    Tensor
        Resultant (batch) state fidelity
    """
    
    sqrt_a = msqrt_herm(a)

    return (msqrt_herm(sqrt_a @ b @ sqrt_a)).diagonal(dim1=-2, dim2=-1).real ** 2
