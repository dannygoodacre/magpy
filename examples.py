from magpy import X, Y, Z, I, set_print_identities, set_print_precision, FunctionProduct as FP
from torch import sin, cos

# Defaults to false
set_print_identities(True)

# Defaults to full precision.
# Also sets torch's print options. Will need to look at this for numpy, qutip, etc., too.
set_print_precision(3)

# PauliString

P = (1, 5.6868 + 1.2345j)*X()
Q = 3*Y(3)

# HamOp

# I can't think of a nicer way of structuring function products than this.
# The leading FP() is needed to 'kickstart' the product chain.
H = FP()*(1,2)*sin*X()
G = cos*Y(2)
