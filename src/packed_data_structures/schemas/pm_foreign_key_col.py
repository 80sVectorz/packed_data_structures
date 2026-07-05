from __future__ import annotations
from packed_data_structures.schemas.injected_logic_col import InjectedLogicColSchema
from packed_data_structures.schemas import IndexSpec
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

import numpy as np

from packed_data_structures.arrays import PackedArray
from .col_schema_like import ColSchemaLike
from .foreign_key_col import AdjacencyListConf, FksOnDeleteStyle

from .adjacency_columns import (
    AdjHeadIdxColSchema,
    AdjNextIdxColSchema,
    AdjPrevIdxColSchema,
    AdjCountColSchema,
)


if TYPE_CHECKING:
    from .table import TableSchema

BITS_TO_UINT_DTYPE = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint8,
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}


class PmFkTypeIdColSchema[T: np.unsignedinteger[Any]](InjectedLogicColSchema[T]):
    @property
    def allows_updates(self):
        return True


@dataclass(eq=False)
class PolymorphicForeignKeySchema[
    T_parent: np.integer[Any],
    T_counts: np.integer[Any],
](ColSchemaLike[np.integer[Any]]):
    """Schema for a column of polymorphic foreign keys.

    Each key can link to one of a set of tables.
    Just like the non-polymorphic FK schema, it automatically injects adjacency list columns:
    - `adj_head` (in target table)
    - `adj_next` (in source table)
    - `adj_prev` (in source table)
    - `adj_count` (optional, in target table)

    NOTE: The true column names are more verbose

    It also injects a `_type_id` column, this column encodes which table each key points to.
    The smallest possible uint dtype is chosen for the `_type_id` columm.

    Type Parameters:
        T_parent: The index dtype of the parent table. And by extension the `adj_next` and `adj_prev` columns.
        T_counts: The dtype of the counts column (if applicable).

    Attributes:
        name: The string identifier for the column.
        target_tables: The ordered collection of TableSchemas that the keys can point to.
        on_delete: The policy to apply when something tries to delete a row referenced by this column.
        adjacency_conf: Configuration for the injected adjacency list columns.
        keys_idx_spec: IndexSpec that holds the dtype and sentinel for the keys.
            Optmized to be only as big as needed to link into any row of the target tables.
        type_id_dtype: The dtype used for the `_type_id` column.
        type_id_sentinel: The sentinel value for the `_type_id` column.
        type_id_mapping: Mapping from table to its type id.
            This is just a direct mapping to the index of each target in `target_tables`
        type_id_col: The injected `_type_id` column.
        adj_next: The injected next-pointer column in the parent table.
        adj_prev: The injected previous-pointer column in the parent table.
        adj_head_columns: The injected head-pointer columns in the target tables, ordered by `type_id`.
        adj_count_columns: The injected count columns in the target table (if enabled).
            Also ordered by `type_id`
    """

    name: str
    target_tables: Sequence[TableSchema]

    on_delete: FksOnDeleteStyle = FksOnDeleteStyle.CASCADE
    adjacency_conf: AdjacencyListConf[T_counts] = field(
        default_factory=AdjacencyListConf
    )

    keys_idx_spec: IndexSpec[np.unsignedinteger[Any]] = field(init=False)

    type_id_dtype: type[np.unsignedinteger[Any]] = field(init=False)
    type_id_sentinel: int = field(init=False)
    type_id_col: PmFkTypeIdColSchema[np.unsignedinteger[Any]] = field(init=False)
    type_id_mapping: dict[TableSchema, int] = field(init=False)

    adj_next: AdjNextIdxColSchema[T_parent] = field(init=False)
    adj_prev: AdjPrevIdxColSchema[T_parent] = field(init=False)
    adj_head_columns: list[AdjHeadIdxColSchema[T_parent]] = field(init=False)
    adj_count_columns: list[AdjCountColSchema[T_counts]] = field(init=False)

    def __post_init__(self):
        n_targets = len(self.target_tables)

        optimal_power = n_targets.bit_length()
        self.type_id_dtype = BITS_TO_UINT_DTYPE[optimal_power]
        self.type_id_sentinel = np.iinfo(self.type_id_dtype).max

        self.type_id_mapping = {
            target: i for i, target in enumerate(self.target_tables)
        }

        required_max_value = 0

        for target in self.target_tables:
            required_max_value = max(required_max_value, target.index_spec.max_value)
            target.subscribe(self)

        self.keys_idx_spec = IndexSpec.from_dtype(
            BITS_TO_UINT_DTYPE[required_max_value.bit_length()]
        )

    def set_parent(self, parent: TableSchema[T_parent]):
        """Bind this polymorphic foreign key column to a parent table.

        This actively mutates both the parent and target table schemas by
        registering the hidden tag column and adjacency list management columns.

        Args:
            parent: The TableSchema that will own this foreign key.
        """
        super().set_parent(parent)

        targets = self.target_tables
        parent = self.parent_table

        full_name = f"{parent.name}_{self.name}"

        self.type_id_col = PmFkTypeIdColSchema(
            f"_type_id_{full_name}",
            self.type_id_dtype,
            self.type_id_sentinel,
        ).set_parent_col(self)

        self.adj_head_columns = [
            AdjHeadIdxColSchema(
                f"_adj_head_{full_name}",
                parent.index_spec.dtype,
                parent.index_spec.missing,
            ).set_parent_col(self)
            for _ in targets
        ]
        self.adj_next = AdjNextIdxColSchema(
            f"_adj_next_{full_name}",
            parent.index_spec.dtype,
            parent.index_spec.missing,
        ).set_parent_col(self)
        self.adj_prev = AdjPrevIdxColSchema(
            f"_adj_prev_{full_name}",
            parent.index_spec.dtype,
            parent.index_spec.missing,
        ).set_parent_col(self)

        for target, adj_head in zip(targets, self.adj_head_columns, strict=True):
            target.register_new_column(adj_head)

        parent.register_new_column(self.type_id_col)
        parent.register_new_column(self.adj_next)
        parent.register_new_column(self.adj_prev)

        if self.adjacency_conf.track_counts:
            assert self.adjacency_conf.counts_dtype is not None
            self.adj_count_columns = []

            for target in targets:
                adj_count = AdjCountColSchema(
                    f"_adj_count_{full_name}",
                    self.adjacency_conf.counts_dtype,
                    0,
                )
                target.register_new_column(adj_count)
                self.adj_count_columns.append(adj_count)

    def __hash__(self) -> int:
        return hash(id(self))

    def init_array(
        self,
    ) -> PackedArray:
        return PackedArray(
            self.parent_table.pre_allocate,
            self.keys_idx_spec.dtype,
            self.keys_idx_spec.missing,
        )
