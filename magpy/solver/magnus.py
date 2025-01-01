"""The Magnus expansion for the solution to the Liouville-von Neumann equation.


References
----------

.. [1] Magnus, W. (1954), "On the exponential solution of differential
       equations for a linear operator", *Comm. Pure Appl. Math.* 7, 649-673.

"""

import torch
from .gauss_legendre_quadrature import get_knots_over_interval, compute_integrals, compute_double_integral_of_commutator


def first_term(H, tlist, n_qubits):
    """Calculate the first term of the Magnus expansion over each given
    interval.

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator
    tlist : Tensor
        Time grid
    n_qubits : int
        Number of qubits

    Returns
    -------
    Tensor
        The first term of the Magnus expansion over each given interval
    """
    step = tlist[1] - tlist[0]
    knots = get_knots_over_interval(tlist, step)
    function_integrals = compute_integrals(H.funcs(), knots, step)
    constant_operators = [op(n_qubits) for op in H.pauli_operators()]

    return torch.tensordot(function_integrals, torch.stack(constant_operators), dims=([0], [0]))


def second_term(H, tlist, n_qubits):
    """Calculate the second term of the Magnus expansion over each given
    interval.

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator
    tlist : Tensor
        Time grid
    n_qubits : int
        Number of qubits

    Returns
    -------
    Tensor
        The second term of the Magnus expansion over each given interval
    """
    funcs = H.funcs()
    pauli_operator_values = torch.stack([op(n_qubits) for op in H.pauli_operators()])

    return compute_double_integral_of_commutator(funcs, pauli_operator_values, tlist, n_qubits)
