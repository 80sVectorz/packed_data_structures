from __future__ import annotations
from typing import Any, TYPE_CHECKING
from packed_data_structures.nb_adjacency_list_helpers import (
    nb_count_adj_elements,
    nb_get_adj_elements,
)
from collections.abc import Iterable

if TYPE_CHECKING:
    from packed_data_structures import PackedArrayTable
from .schema_accessor import SchemaAccessor
from packed_data_structures.schemas import ForeignKeySchema
from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class ForeignKeySchemaAccessor[
    T: np.integer[Any],
    T_parent: np.integer[Any],
    T_counts: np.integer[Any],
](SchemaAccessor[ForeignKeySchema[T, T_parent, T_counts], T]):
    """Provides a typed view into a foreign key column.

    Returned when indexing a table with a ForeignKeySchema. In addition to
    raw array access, it provides methods for querying adjacency lists.

    Attributes:
        table: The table containing the foreign key.
        schema: The foreign key schema definition.
        col_id: The internal array index for this column.
        target_table: The table this foreign key points to.
    """

    table: PackedArrayTable
    schema: ForeignKeySchema[T, T_parent, T_counts]
    col_id: int

    target_table: PackedArrayTable[T] = field(init=False)

    def __post_init__(self):
        self.target_table = self.table.db.get_table(self.schema.target_table)

    def get_referencing_indices(
        self, head_indices: Iterable[int]
    ) -> tuple[tuple[int, ...], ...]:
        """Fetch the list of row indices that reference given targets.

        Uses the dynamically injected adjacency list columns to traverse
        relationships backwards in O(K) time, where K is the number of incident edges.

        Args:
            head_indices: The target row indices to query incoming links for.

        Returns:
            A tuple containing a tuple of referring row indices for each target.
        """
        head_indices = tuple(head_indices)

        vw_adj_head = self.target_table[self.schema.adj_head].view
        vw_adj_next = self.table[self.schema.adj_next].view

        index_spec = self.table.schema.index_spec
        if not self.schema.adjacency_conf.track_counts:
            counts = np.zeros_like(head_indices, index_spec.dtype)
            nb_count_adj_elements(
                head_indices, vw_adj_head, vw_adj_next, index_spec.missing, out=counts
            )

        else:
            vw_adj_count = self.target_table[self.schema.adj_count].view
            counts = vw_adj_count[list(head_indices)]

        indices = tuple(np.zeros(cnt, index_spec.dtype) for cnt in counts)

        nb_get_adj_elements(head_indices, counts, vw_adj_head, vw_adj_next, out=indices)

        return tuple(tuple(v.astype(int)) for v in indices)
