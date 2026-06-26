from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from packed_data_structures.transaction_context import TransactionContext
from packed_data_structures.schemas import TableSchema


@runtime_checkable
class TransactionHook(Protocol):
    """Interface for injecting logic into the database transaction lifecycle."""

    def on_transaction_start(self, ctx: TransactionContext) -> None:
        """Called when a transaction context is entered."""
        ...

    def on_register_additions(
        self, ctx: TransactionContext, table: TableSchema, indices: range
    ) -> None:
        """Called immediately after rows are added to the staging area."""
        ...

    def before_commit_planning(
        self, ctx: TransactionContext
    ) -> dict[str, dict[int, dict[str, Any]]] | None:
        """The 'Compute Phase'. Called before the bulk edit plan is generated.

        Return a dictionary of updates:
            {TableName: {RowID: {ColName: Value}}}

        RowID can be a Staged ID (for rows just added). The DB will apply
        these updates to the staging buffer before committing.
        """
        ...

    def after_commit(self, ctx: TransactionContext) -> None:
        """Called after all changes are finalized and written to memory."""
        ...
