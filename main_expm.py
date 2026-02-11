import magpy as mp
from numbers import Number
from magpy import PauliString, HamOp as HOp
import torch
from torch import Tensor

def expm(H: HOp, t):
    return mp.kron(*[(coeff*op.as_single_qubit()).expm().matrix() for coeff, op in H.unpack(t, unit_ops=True)])
