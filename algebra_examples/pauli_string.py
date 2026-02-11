"""
The most basic of operators. A product of pauli qubits and a constant coefficient (that may be batched).
"""

from magpy import X, Y, Z, I, set_print_identities, set_print_precision

# default = full precision
set_print_precision(3)

# default = false
set_print_identities(True)

P = 3*X()

print(P)

Q = (2, 5 + 1.2345j)*Y(1)

print(Q)
