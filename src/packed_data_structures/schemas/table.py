from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from packed_data_structures.arrays import PackedArray
from .index_spec import IndexSpec

if TYPE_CHECKING:
    from .col_schema_like import ColSchemaLike
    from .foreign_key_col import ForeignKeySchema


class SupportsGetTableSchema[T_idx: np.integer[Any]](ABC):
    """Interface for objects that can provide a TableSchema."""

    @abstractmethod
    def get_table_schema(self) -> TableSchema[T_idx]:
        """Get the underlying TableSchema.

        Returns:
            The TableSchema instance.
        """
        ...


@dataclass(slots=True)
class TableSchema[T_idx: np.integer[Any]](SupportsGetTableSchema[T_idx]):
    """Schema definition for a flat, column-oriented table.

    A TableSchema aggregates multiple `ColSchemaLike` definitions and
    dictates how the `PackedArrayTable` initializes its raw buffers.

    Attributes:
        name: The string identifier for the table.
        index_spec: The specification defining the table's index type and capacity.
        cols: The list of column schemas defining the table structure.
        pre_allocate: The initial element capacity to allocate for the table's arrays.
    """

    name: str
    index_spec: IndexSpec[T_idx]
    cols: list[ColSchemaLike[Any]]
    pre_allocate: int = 0

    subscribers: list[ForeignKeySchema[Any, T_idx, Any]] = field(
        init=False, default_factory=list
    )
    col_ids: dict[ColSchemaLike, int] = field(init=False, default_factory=dict)
    _finalized: bool = field(init=False, default=False)

    def __post_init__(self):
        for col in self.cols:
            col.set_parent(self)

        for i, col in enumerate(self.cols):
            self.col_ids[col] = i

    def subscribe(self, new_subscriber: ForeignKeySchema):
        """Register a foreign key that targets this table.

        Args:
            new_subscriber: The foreign key schema pointing to this table.
        """
        if new_subscriber not in self.subscribers:
            self.subscribers.append(new_subscriber)

    def init_arrays(
        self,
    ) -> tuple[PackedArray, ...]:
        self._finalized = True
        return tuple(col.init_array() for col in self.cols)

    def register_new_column(self, col: ColSchemaLike):
        """Dynamically add a new column to the table schema.

        This is primarily used by overlay features and foreign keys to inject hidden
        management columns prior to initialization.

        Args:
            col: The column schema to add.

        Raises:
            RuntimeError: If the schema has already been initialized.
        """
        if self._finalized:
            raise RuntimeError(
                f"Cannot register column '{col.name}' to table '{self.name}': "
                "Schema is already finalized/initialized"
            )

        self.col_ids[col] = len(self.cols)

        self.cols.append(col)
        col.set_parent(self)

    def get_table_schema(self) -> TableSchema:
        return self
