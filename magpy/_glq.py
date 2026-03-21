from math import sqrt

import torch

from magpy._context import _CONTEXT


_KNOTS_3 = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)
_WEIGHTS_3 = torch.tensor([5/9, 8/9, 5/9], dtype=torch.complex128)


def get_knots_3():
    return _KNOTS_3.to(_CONTEXT.device)


def get_weights_3():
    return _WEIGHTS_3.to(_CONTEXT.device)
