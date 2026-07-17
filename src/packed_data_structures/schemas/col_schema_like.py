from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from packed_data_structures.arrays import PackedArray

if TYPE_CHECKING:
    from .table import TableSchema


@dataclass(eq=False)
class ColSchemaLike[T: np.generic](ABC):
    """Base class for all column schemas.

    Column schemas are treated as object-identity singletons (`id(self)`)
    when used as dictionary keys or accessors. They define a single column
    within a TableSchema.

    Attributes:
        name: The string identifier for the column.
        parent_table: The TableSchema instance this column belongs to.
    """

    name: str
    parent_table: TableSchema = field(init=False)

    def set_parent(self, parent: TableSchema):
        """Bind this column to a parent TableSchema.

        Args:
            parent: The TableSchema that will own this column.
        """
        self.parent_table = parent

    @abstractmethod
    def init_array(self) -> PackedArray[T]: ...

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other
