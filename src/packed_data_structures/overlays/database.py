from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field

from packed_data_structures.table import PackedArrayTable
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.transaction_context import TransactionContext
from packed_data_structures.schemas import TableSchema
from .registry import SchemaRegistry
from .base import DbOverlay
from .hooks import TransactionHook


@dataclass(init=False)
class OverlaidDB(PackedArrayDB):
    """A PackedArrayDB that builds itself from Overlays."""

    table_ids: dict[str, int] = field(init=False)
    table_schemas: tuple[TableSchema, ...] = field(init=False)
    tables: tuple[PackedArrayTable, ...] = field(init=False)

    _transaction_ctx: TransactionContext | None = field(init=False, default=None)

    overlays: tuple[DbOverlay, ...]

    def __init__(self, *overlays: DbOverlay):
        self.overlays = overlays

        # Compile Schema from all Overlays
        registry = SchemaRegistry()
        for o in overlays:
            o.register_schema(registry)

        schemas = registry.build()

        super().__init__(*schemas)

        # Bind Overlays to the runtime
        for o in overlays:
            o.bind(self)

    def transaction(self) -> TransactionContext:
        ctx = self._transaction_ctx
        if ctx is None:
            ctx = _OverlaidTransactionContext(self)
            self._transaction_ctx = ctx
        return ctx


@dataclass()
class _OverlaidTransactionContext(TransactionContext):
    """Enhanced Context that delegates lifecycle events to Hooks."""

    hooks: list[TransactionHook] = field(init=False, default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Collect hooks from all registered overlays
        if isinstance(self.db, OverlaidDB):
            for o in self.db.overlays:
                self.hooks.extend(o.create_hooks())

        # Notify Start
        for h in self.hooks:
            if hasattr(h, "on_transaction_start"):
                h.on_transaction_start(self)

    def register_additions_col_major(self, table: TableSchema, values: Any) -> range:
        # Standard Core Logic
        new_ids = super().register_additions_col_major(table, values)

        # Notify Hooks
        for h in self.hooks:
            if hasattr(h, "on_register_additions"):
                h.on_register_additions(self, table, new_ids)
        return new_ids

    def commit(self):
        # Pre-Commit Planning
        # Allow hooks to compute derived data based on the staging area
        updates_buffer = []
        for h in self.hooks:
            if hasattr(h, "before_commit_planning"):
                updates = h.before_commit_planning(self)
                if updates:
                    updates_buffer.append(updates)

        # Apply derived updates directly to the staging buffers
        # This effectively "patches" the new rows before they are committed.
        for update_batch in updates_buffer:
            for tbl_name, rows in update_batch.items():
                if tbl_name not in self.additions:
                    continue

                table = self.db.get_table(tbl_name)
                # Calculate the start of the virtual ID range for the current buffer
                buf_len = len(self.additions[tbl_name][0])
                virt_start = self._virtual_counters[tbl_name] - buf_len

                for row_id, col_map in rows.items():
                    # We only patch Virtual IDs (new rows) here
                    if row_id >= virt_start:
                        offset = row_id - virt_start
                        for col_name, val in col_map.items():
                            col_idx = table.column_ids[col_name]
                            self.additions[tbl_name][col_idx][offset] = val

        # Core Commit
        # Hand off to the robust swizzling/adjacency logic in table.py
        super().commit()

        # Post-Commit
        for h in self.hooks:
            if hasattr(h, "after_commit"):
                h.after_commit(self)
