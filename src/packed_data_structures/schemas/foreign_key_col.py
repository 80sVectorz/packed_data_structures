from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from packed_data_structures.arrays import PackedArray
from .col_schema_like import ColSchemaLike
from .data_col import DataColSchema

if TYPE_CHECKING:
    from .table import TableSchema


@dataclass
class AdjacencyListConf[T_counts: np.generic]:
    """Config for adjacency list structure.

    Attributes:
        track_counts: If a counts column should be made.
        counts_dtype: Dtype of the element count column. Defaults to np.uint8
    """

    track_counts: bool = False
    counts_dtype: type[T_counts] | None = None


class FksOnDeleteStyle(Enum):
    """Various ForeignKeySchema on referenced row delete behaviors."""

    CASCADE = auto()
    """Cascade the deletion by also deleting the FK row"""
    RESTRICT = auto()
    """Block the deletion and raise an exception"""
    SET_NULL = auto()
    """Set FK field to missing index value"""


@dataclass(eq=False)
class ForeignKeySchema[
    T: np.integer[Any],
    T_parent: np.integer[Any],
    T_counts: np.integer[Any],
](ColSchemaLike[T]):
    """Schema for a foreign key relationship linking to another table.

    When registered, this schema dynamically injects internal adjacency list
    columns (`adj_head`, `adj_next`, `adj_prev`, and optionally `adj_count`)
    into both the source and target tables.

    Attributes:
        name: The string identifier for the foreign key.
        target_table: The TableSchema of the table this key points to.
        on_delete: The policy to apply when the referenced row is deleted.
        adjacency_conf: Configuration for the injected adjacency list columns.
        adj_next: The injected next-pointer column in the parent table.
        adj_prev: The injected previous-pointer column in the parent table.
        adj_head: The injected head-pointer column in the target table.
        adj_count: The injected count column in the target table (if enabled).
    """

    name: str
    target_table: TableSchema
    on_delete: FksOnDeleteStyle = FksOnDeleteStyle.CASCADE
    adjacency_conf: AdjacencyListConf[T_counts] = field(
        default_factory=AdjacencyListConf
    )

    adj_next: DataColSchema[T_parent] = field(init=False)
    adj_prev: DataColSchema[T_parent] = field(init=False)
    adj_head: DataColSchema[T] = field(init=False)
    adj_count: DataColSchema[T_counts] = field(init=False)

    def __post_init__(self):
        self.target_table.subscribe(self)

    def set_parent(self, parent: TableSchema[T_parent]):
        """Bind this foreign key to a parent table and inject adjacency columns.

        This actively mutates both the parent and target table schemas by
        registering the hidden adjacency list management columns.

        Args:
            parent: The TableSchema that will own this foreign key.
        """
        super().set_parent(parent)

        target = self.target_table
        parent = self.parent_table

        full_name = f"{parent.name}_{self.name}"

        self.adj_head = DataColSchema(
            f"_adj_head_{full_name}",
            parent.index_spec.dtype,
            parent.index_spec.missing,
        )
        self.adj_next = DataColSchema(
            f"_adj_next_{full_name}",
            parent.index_spec.dtype,
            parent.index_spec.missing,
        )
        self.adj_prev = DataColSchema(
            f"_adj_prev_{full_name}",
            parent.index_spec.dtype,
            parent.index_spec.missing,
        )

        target.register_new_column(self.adj_head)
        parent.register_new_column(self.adj_next)
        parent.register_new_column(self.adj_prev)

        if self.adjacency_conf.track_counts:
            assert self.adjacency_conf.counts_dtype is not None
            self.adj_count = DataColSchema(
                f"_adj_count_{full_name}",
                self.adjacency_conf.counts_dtype,
                0,
            )
            target.register_new_column(self.adj_count)

    def __hash__(self) -> int:
        return hash(id(self))

    def init_array(
        self,
    ) -> PackedArray:
        return PackedArray(
            self.parent_table.pre_allocate,
            self.target_table.index_spec.dtype,
            self.target_table.index_spec.missing,
        )
