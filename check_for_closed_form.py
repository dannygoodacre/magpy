from magpy import X, Y, Z, PauliString, HamiltonianOperator
from torch import tensor

H = tensor([1,2])*X() + 3*Y()

def check_for_closed_form(H: PauliString | HamiltonianOperator) -> bool:

    if isinstance(H, PauliString):
        return True

print(H.pauli_operators())