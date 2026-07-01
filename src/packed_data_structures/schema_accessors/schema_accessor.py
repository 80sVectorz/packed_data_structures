from __future__ import annotations
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from packed_data_structures import PackedArrayTable, PackedArray
from packed_data_structures.schemas import ColSchemaLike
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SchemaAccessor[
    T_s: ColSchemaLike,
    T: np.generic,
    *T_shape,
]:
    """Provides a typed view into a specific column of a table.

    Returned when indexing a table with a DataColSchema. Provides
    direct interaction with the underlying numpy arrays.

    Attributes:
        table: The table containing the data.
        schema: The schema definition for this column.
        col_id: The internal array index for this column.
    """

    table: PackedArrayTable
    schema: T_s
    col_id: int

    @property
    def view(self) -> np.ndarray[tuple[int, *T_shape], np.dtype[T]]:
        """Access the raw numpy view of the packed array.

        This provides direct, zero-overhead access to the contiguous array
        backing this column, suitable for passing to Numba or vectorized
        Numpy operations.

        Returns:
            A numpy array view of the active elements.
        """
        return cast(
            np.ndarray[tuple[int, *T_shape], np.dtype[T]],
            self.table.arrays[self.col_id].view,
        )

    def __getitem__(self, *args):
        return self.view.__getitem__(*args)

    @property
    def arr(self) -> PackedArray[T, *T_shape]:  # ty: ignore[invalid-type-arguments]
        return self.table.arrays[self.col_id]
