from magpy import X, Y, Z, FunctionProduct as FP, HamiltonianOperator
from torch import tensor, complex128, sin, cos, pi, matrix_exp

H = HamiltonianOperator((sin, X()))

t = tensor([1,2], dtype=complex128)

print(H)
print(H(t))
print(H.propagator(t))
print(H.propagator(t).matrix())

print(matrix_exp(-1j*H(t).matrix()))
