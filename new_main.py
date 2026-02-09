from magpy import PauliString, X, Y, Z
from magpy import print_identities, set_print_precision, FunctionProduct as FP, propagator

from torch import tensor, complex128, cos, sin

set_print_precision(3)

p = (1.23948,2.48945894,3.3495845)*X() + 5.983*Y()*Z(1)

print(p)
print(p.matrix())
