from numbers import Number
from typing import TypeAlias

from torch import Tensor

Scalar: TypeAlias = Number | Tensor | list | tuple
