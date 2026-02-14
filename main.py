from magpy import FunctionProduct as FP, X, Y, timegrid, frob, new_evolve
from torch import sin, sqrt, tensor
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


H = (1,2)*X()

print(H*Y())
