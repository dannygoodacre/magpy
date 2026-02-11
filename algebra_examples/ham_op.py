"""
Any linear combination of Pauli strings, and/or any with a functional coefficient.
"""

from magpy import X, Y, Z, I
from torch import sin

H = 3*X() + (1,2)*Y()

print(H)

G = sin*Z(2)

print(G)
