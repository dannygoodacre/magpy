from magpy import FunctionProduct as FP, X, Y, timegrid, frob, evolve, set_print_precision, set_print_identities
from torch import sin, sqrt, tensor
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

set_print_identities(True)

set_print_precision(3)

H = sin * Y(1)

print(H(tensor([3,4])).tensor())
