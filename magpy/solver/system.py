"""Evolve the density matrix under the specified Hamiltonian, starting
from the initial condition.

"""

import torch
from ..core import PauliString, HamiltonianOperator
from .magnus import first_term, second_term

def evolve(H: HamiltonianOperator, rho0: PauliString, tlist: torch.Tensor, n_qubits: int = None) -> torch.Tensor:
    """Evolve the density matrix under the specified Hamiltonian, starting
    from the initial condition.

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator.
    rho0 : PauliString
        The initial density matrix.
    tlist : Tensor
        The list of times at which to return the state.
    n_qubits : int
        The number of qubits.

    Returns
    -------
    torch.Tensor
        The state at each time in tlist.
    """

    if isinstance(H, PauliString):
        H = HamiltonianOperator([1, H])

    if n_qubits is None:
        n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators())

    states = torch.empty((len(tlist), 2 ** n_qubits, 2 ** n_qubits), dtype=torch.complex128)
    states[0] = rho0()

    if H.is_constant():
        return _evolve_time_independent(H, tlist, states)
    return _evolve_time_dependent(H, tlist, n_qubits, states)

def _evolve_time_independent(H: HamiltonianOperator, tlist: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Evolve the density matrix under a time-independent Hamiltonian.

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator.
    tlist : Tensor
        Time grid.
    states : Tensor
        Empty state tensor in which to store the result.

    Returns
    -------
    Tensor
        The evolved state at each time in `tlist`.
    """
    # Calculate constant propagators.
    u = torch.matrix_exp(-1j * (tlist[1] - tlist[0]) * H())
    ut = torch.conj(torch.transpose(u, 0, 1))

    # Evolve the state in series.
    for i in range(len(tlist) - 1):
        states[i + 1] = u @ states[i] @ ut

    return states

def _evolve_time_dependent(H: HamiltonianOperator, tlist: torch.Tensor, n_qubits: int, states: torch.Tensor) \
        -> torch.Tensor:
    """Evolve the density matrix under a time-dependent Hamiltonian.

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator.
    tlist : Tensor
        Time grid.
    n_qubits : int
        The number of qubits.
    states : Tensor
        Empty state tensor in which to store the result.

    Returns
    -------
    Tensor
        The evolved state at each time in `tlist`.
    """
    # Calculate expansion and propagators in parallel.
    omega1 = first_term(H, tlist, n_qubits)
    omega2 = second_term(H, tlist, n_qubits)
    u = torch.matrix_exp(-1j * (omega1 + omega2))
    ut = torch.conj(torch.transpose(u, 1, 2))

    # Evolve the state in series.
    for i in range(len(tlist) - 1):
        states[i + 1] = u[i] @ states[i] @ ut[i]

    return states
