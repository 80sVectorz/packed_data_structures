from __future__ import annotations
from packed_data_structures.schemas.object_col import ObjectColSchema
from packed_data_structures.arrays import PackedObjectArray

from typing import Any, cast, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from packed_data_structures import PackedArrayTable
from .schema_accessor import SchemaAccessor
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ObjectColSchemaAccessor[
    T: Any,
](SchemaAccessor[ObjectColSchema[T], T]):
    """Provides a typed view into an object column.

    Returned when indexing a table with an ObjectColSchema.

    Attributes:
        table: The table containing the object array.
        schema: The object column schema definition.
        col_id: The internal array index for this column.
    """

    table: PackedArrayTable
    schema: ObjectColSchema[T]
    col_id: int
    array: PackedObjectArray[T] = field(init=False)

    def __post_init__(self):
        self.array = cast(PackedObjectArray[T], self.table.arrays[self.col_id])

    @property
    def view(self) -> np.ndarray[tuple[int], np.dtype[np.object_]]:
        """Access the raw numpy view of the packed array.

        This provides direct, zero-overhead access to the contiguous array
        backing this column, suitable for passing to Numba or vectorized
        Numpy operations.

        Returns:
            A numpy array view of the active elements.
        """
        return cast(
            np.ndarray[tuple[int], np.dtype[np.object_]],
            self.table.arrays[self.col_id].view,
        )

    @overload
    def __getitem__(
        self, arg: slice | np.ndarray[tuple[int], np.dtype[np.integer[Any]]]
    ) -> np.ndarray[tuple[int], np.dtype[np.object_]]: ...

    @overload
    def __getitem__(self, arg: int | np.integer[Any]) -> T: ...

    def __getitem__(
        self, arg: int | np.integer[Any] | slice | np.ndarray
    ) -> T | np.ndarray[tuple[int], np.dtype[np.object_]]:  # ty:ignore[invalid-method-override]
        return self.view.__getitem__(arg)

    @property
    def arr(self) -> PackedObjectArray[T]:
        return self.array
