from magpy import X, Y, Z, FunctionProduct as FP, HamiltonianOperator
from torch import tensor, complex128, sin, cos, pi, matrix_exp

H = sin*X()

h = 0.1
t = tensor([1,2], dtype=complex128)

# print(H)
# print(H(t))
# print(H.propagator(h, t))
print(H.propagator(h, t).matrix())

print(matrix_exp(-1j*h*H(t).matrix()))
