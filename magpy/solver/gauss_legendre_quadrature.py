"""Gauss-Legendre quadrature methods, degree 3.

This module implements the methods described by Iserles et al.

References
----------

.. [1] Iserles, A., Munthe-Kaas, H. Z., Nørsett, S. P. & Zanna, A. (2000),
       "Lie-group methods", *Acta Numerica* 9, 215-365.

"""

import torch
from math import sqrt
from .._device import _DEVICE_CONTEXT

_QUADRATURE_DEGREE = 3
_KNOTS = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)

_WEIGHTS_1 = torch.tensor([5/9, 8/9, 5/9])

_WEIGHTS_2 = torch.tensor([2, 1, 2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
_WEIGHTS_2_COEFFICIENT = sqrt(15) / 54

_combinations = [(0, 1), (0, 2), (1, 2)]


def _update_device():
    global _KNOTS, _WEIGHTS_1, _WEIGHTS_2
    _KNOTS = _KNOTS.to(_DEVICE_CONTEXT.device)
    _WEIGHTS_1 = _WEIGHTS_1.to(_DEVICE_CONTEXT.device)
    _WEIGHTS_2 = _WEIGHTS_2.to(_DEVICE_CONTEXT.device)


def get_knots_over_interval(tlist: torch.Tensor, step: float) -> torch.Tensor:
    """Translate GLQ knots to the given intervals.

    Given `n` points in time, this function returns a tensor with shape
    `[n-1, 3]`.

    Parameters
    ----------
    tlist : Tensor
         Time grid.
    step : float
        Time step.

    Returns
    -------
    Tensor
        Knots over each grid interval
    """
    midpoints = tlist[0] + step*(torch.arange(len(tlist) - 1) + 0.5)
    return 0.5*step*_KNOTS + midpoints[:, None]


def compute_integrals(funcs: list, knots: torch.Tensor, step: float) -> torch.Tensor:
    """Compute the integrals of the given functions over the intervals
    determined by the given knots, using GLQ3.

    Given `m` functions and `n-1` intervals (from `n` points in time), this
    function returns a tensor with shape `[m, n-1]`.

    Parameters
    ----------
    funcs : list[function]
        Functions.
    knots : Tensor
        Knots over each grid interval.
    step : float
        Time step.

    Returns
    -------
    Tensor
        Integrals of the given functions over the given intervals.
    """
    weighted_functions = tuple(torch.ones(knots.shape) * _WEIGHTS_1
                               if f == 1 else f(knots) * _WEIGHTS_1 for f in funcs)
    return 0.5 * step * torch.sum(torch.stack(weighted_functions), 2)


def compute_double_integral_of_commutator(funcs: list, pauli_op_matrices: torch.Tensor, tlist: torch.Tensor,
                                          n_qubits: int) -> torch.Tensor:
    """Compute the double integral of the commutator of the Hamiltonian
    specified by the given functions and Pauli operators.

    Parameters
    ----------
    funcs : list[function]
        Coefficient function of the Hamiltonian.
    pauli_op_matrices : Tensor
        Pauli operator matrices of the Hamiltonian.
    tlist : Tensor
        Time grid.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    Tensor
        Double integral of the commutator of the Hamiltonian over the
        given intervals.
    """
    n = len(tlist) - 1
    step = tlist[1] - tlist[0]
    dim = 2 ** n_qubits
    knots = get_knots_over_interval(tlist, step)
    commutators = torch.stack([_eval_commutator(i, j, n, step, dim, knots, funcs, pauli_op_matrices)
                               for i, j in _combinations])

    return _WEIGHTS_2_COEFFICIENT * torch.sum(commutators * _WEIGHTS_2, 0)


def _eval_commutator(i: int, j: int, n: int, step: float, dim: int, knots: torch.Tensor, funcs: list,
                     pauli_op_matrices: torch.Tensor) -> torch.Tensor:
    """Evaluate the commutator of the Hamiltonian ...

    Parameters
    ----------
    i : int
        Index of the first column.
    j : int
        Index of the second column.
    n : int
        Number of intervals.
    step : float
        Time step.
    dim : int
        Dimension of the system
    knots : Tensor
        Knots over each grid interval.
    funcs : list[function]
        Coefficient function of the Hamiltonian.
    pauli_op_matrices : Tensor
        Pauli operator matrices of the Hamiltonian.

    Returns
    -------
    Tensor
        Commutator of the Hamiltonian over the given intervals.
    """
    slices = _slices(knots, i, j)
    op_vals_outer_product = pauli_op_matrices.unsqueeze(1) @ pauli_op_matrices.unsqueeze(0)
    func_vals = _evaluate_funcs(funcs, slices)
    func_vals_outer_product = _compute_outer_products(func_vals)

    return (step ** 2) * torch.sum(
        (func_vals_outer_product * (op_vals_outer_product - op_vals_outer_product.transpose(0, 1)))
        .reshape((n, len(funcs) ** 2, dim, dim)), 1)


def _slices(knots: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Take the `i`-th and `j`-th columns of the given knots.

    Parameters
    ----------
    knots : Tensor
        Knots over each grid interval.
    i : int
       Index of the first column.
    j : int
        Index of the second column.

    Returns
    -------
    Tensor
        Slices of the given knots.
    """
    return torch.stack((knots[:, i], knots[:, j])).transpose(0, 1)


def _evaluate_funcs(funcs: list, slices: torch.Tensor) -> torch.Tensor:
    """Evaluate the given functions along the given slices.

    Parameters
    ----------
    funcs : list[function]
        Functions.
    slices : Tensor
        Slices of the knots.

    Returns
    -------
    Tensor
        Evaluated functions along the given slices.
    """
    return torch.tensor([[[1 if f == 1 else f(knot) for f in funcs] for knot in knots] for knots in slices])


def _compute_outer_products(v: torch.Tensor) -> torch.Tensor:
    """Evaluate the outer products of the given vectors with themselves.

    Parameters
    ----------
    v : Tensor
        Vectors.

    Returns
    -------
    Tensor
        Outer products of the given vectors with themselves.
    """
    return torch.func.vmap(lambda x: torch.outer(x[0], x[1]))(v).unsqueeze(-1).unsqueeze(-1)
