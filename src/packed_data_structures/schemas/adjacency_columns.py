from __future__ import annotations
from typing import Any

import numpy as np

from .injected_logic_col import InjectedLogicColSchema


class AdjHeadIdxColSchema[T: np.integer[Any]](InjectedLogicColSchema[T]):
    pass


class AdjNextIdxColSchema[T: np.integer[Any]](InjectedLogicColSchema[T]):
    pass


class AdjPrevIdxColSchema[T: np.integer[Any]](InjectedLogicColSchema[T]):
    pass


class AdjCountColSchema[T: np.integer[Any]](InjectedLogicColSchema[T]):
    pass
