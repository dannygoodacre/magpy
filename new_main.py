from magpy import PauliString, X, Y, Z
from magpy import set_print_identities, set_print_precision, FunctionProduct as FP, propagator

from torch import tensor, complex128, cos, sin

set_print_precision(3)
set_print_identities(True)

p = 3*X(0)*Y(1)*Z(3)

print(p)
print()
print(type(p))
print()

p = FP()*3*cos*X()

print(p)
print()
print(type(p))
print()

p = X() + Y()

print(p)
print()
print(type(p))
print()
