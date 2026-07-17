from __future__ import annotations
from typing import Self, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from packed_data_structures.schemas import ColSchemaLike

import numpy as np

from .data_col import DataColSchema


@dataclass(eq=False)
class InjectedLogicColSchema[T: np.generic](DataColSchema[T]):
    parent_col: ColSchemaLike = field(init=False)

    def set_parent_col(self, parent: ColSchemaLike) -> Self:
        self.parent_col = parent
        return self

    @property
    def allows_updates(self):
        return False
