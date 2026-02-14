from math import sqrt

import torch


_KNOTS_3 = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)
_WEIGHTS_3 = torch.tensor([5/9, 8/9, 5/9])
