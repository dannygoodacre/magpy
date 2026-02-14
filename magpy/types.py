from typing import Callable, TypeAlias

from torch import Tensor

Scalar: TypeAlias = int | float | complex | Tensor | list | tuple
Coefficient: TypeAlias = Scalar | Callable

SCALAR_TYPES = int | float | complex | Tensor | list | tuple
