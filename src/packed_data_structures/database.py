from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload, Any
import numpy as np


from packed_data_structures.arrays.dirty_tracking import (
    DirtyTimestampProvider,
    ProvidesDirtyTimestamp,
)
from packed_data_structures.schemas import (
    SupportsGetTableSchema,
    TableSchema,
)
from packed_data_structures.table import PackedArrayTable
from packed_data_structures.transaction_context import TransactionContext


@dataclass(slots=True, init=False)
class PackedArrayDB(ProvidesDirtyTimestamp):
    """The core container managing a collection of PackedArrayTables.

    Acts as the central registry for tables and coordinates transactions
    across them.

    Attributes:
        table_ids: A mapping from table names to their internal indices.
        table_schemas: A tuple of all registered TableSchemas.
        tables: A tuple of the instantiated PackedArrayTables.
    """

    table_ids: dict[str, int] = field(init=False)
    table_schemas: tuple[TableSchema, ...] = field(init=False)
    tables: tuple[PackedArrayTable, ...] = field(init=False)

    _transaction_ctx: TransactionContext | None = field(init=False, default=None)
    _dirty_sources_cache: tuple[DirtyTimestampProvider, ...] | None = field(
        init=False, default=None
    )

    def __init__(self, *tables: TableSchema):
        ProvidesDirtyTimestamp.__init__(self)
        self.table_ids = {t.name: i for i, t in enumerate(tables)}
        self.table_schemas = tables
        self.tables = tuple(PackedArrayTable(self, i, t) for i, t in enumerate(tables))

        self._transaction_ctx = None

    @overload
    def get_table_schema(self, key: str) -> TableSchema: ...

    @overload
    def get_table_schema(self, key: SupportsGetTableSchema) -> TableSchema: ...

    def get_table_schema(self, key: SupportsGetTableSchema | str) -> TableSchema:
        if isinstance(key, SupportsGetTableSchema):
            return key.get_table_schema()

        table_id = self.table_ids.get(key)
        if table_id is None:
            raise KeyError(f"No table named: '{key}'")

        return self.table_schemas[table_id]

    @overload
    def get_table(self, key: str) -> PackedArrayTable: ...

    @overload
    def get_table[T_idx: np.integer[Any]](
        self, key: SupportsGetTableSchema[T_idx]
    ) -> PackedArrayTable[T_idx]: ...

    def get_table[T_idx: np.integer[Any]](
        self, key: SupportsGetTableSchema[T_idx] | str
    ) -> PackedArrayTable[T_idx]:
        if isinstance(key, SupportsGetTableSchema):
            key = key.get_table_schema().name

        table_id = self.table_ids.get(key)
        if table_id is None:
            raise KeyError(f"No table named: '{key}'")
        return self.tables[table_id]

    def transaction(self) -> TransactionContext:
        """Returns a new or existing transaction context."""
        ctx = self._transaction_ctx
        if ctx is None:
            ctx = TransactionContext(self)
            self._transaction_ctx = ctx
        return ctx

    def _transaction_finished(self):
        self._transaction_ctx = None

    # --- dirty tracking ---
    def _collect_dirty_sources(self) -> tuple[DirtyTimestampProvider, ...]:
        return self.tables
