from magpy import PauliString, X, Y, Z
from magpy import print_identities, set_print_precision, FunctionProduct as FP, propagator

from torch import tensor, complex128, cos, sin

set_print_precision(3)

p = sin*X()

# print(p.propagator(tensor([1,2], dtype=complex128)))

print(tensor([1,2j,3,4], dtype=complex128)*X())
