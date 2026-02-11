from magpy import X, Y, Z, I, PauliString, HamOp, set_print_precision, set_print_identities
from torch import tensor
import torch
import qutip as qt

set_print_precision(3)
set_print_identities(True)

def check_for_closed_form(H: PauliString | HamOp) -> bool:

    if isinstance(H, PauliString):
        return True

# TODO: These need formal mathematical documentation!
# I also need to confirm that these work in batches.

# Works for all PauliString instances.
def propagator(P: PauliString, h: float = 1.0) -> HamOp:
    v = h * P.coeff

    return torch.cos(v)*I() - 1j*torch.sin(v)*P.as_unit_operator()

# Works when the PauliStrings in the HamOp are a commuting set.
def propagator1(H: HamOp, h: float = 1.0, t: float = 1.0) -> HamOp:
    result = None


p = X() + Y()
q = Y() + Z()

result = p * q

# print(result.tensor())

print()

# result = (qt.sigmax() + qt.sigmay())*(qt.sigmay() + qt.sigmaz())

# print(result.full())

# This works
print(result)
print(result ** 2)
