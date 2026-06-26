from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

import numpy as np
import numba as nb

from packed_data_structures.edit_helpers import (
    interleave_existing_rows,
    patch_relocated_adjacency_and_heads,
    patch_unlinked_adjacency_and_heads,
    interleave_new_rows,
    plan_bulk_edit,
    fix_keys_broken_by_moved_targets,
)
from packed_data_structures.schemas import (
    ColSchemaLike,
    DataColSchema,
    FksOnDeleteStyle,
    ForeignKeySchema,
    SupportsGetTableSchema,
    TableSchema,
)
from packed_data_structures.table import NormalizedUpdatesColMajor, PackedArrayTable


if TYPE_CHECKING:
    from packed_data_structures.database import PackedArrayDB
    from packed_data_structures.table import (
        NormalizedRecordRowMajor,
        NormalizedRecordsColMajor,
    )


@dataclass(slots=True)
class TopologyScratchpad:
    """A sparse write-buffer for topology updates with cursor-based tracking.

    This scratchpad collects adjacency list patches (next, prev, head pointers)
    during bulk edits before applying them to the physical arrays. This ensures
    that swap-and-pop relocations and unlinks are resolved correctly.
    """

    # Source Table Updates (adj_next / adj_prev)
    next_indices: np.ndarray
    next_values: np.ndarray

    prev_indices: np.ndarray
    prev_values: np.ndarray

    # Target Table Updates (adj_head / adj_count)
    head_indices: np.ndarray
    head_values: np.ndarray

    count_indices: np.ndarray
    count_deltas: np.ndarray

    nullify_indices: np.ndarray

    next_cursor: int = 0
    prev_cursor: int = 0
    head_cursor: int = 0
    count_cursor: int = 0

    @classmethod
    def allocate(
        cls,
        n_unlinks: int,
        n_additions: int,
        n_relinks: int,
        n_moves: int,
        fk_nulls: np.ndarray,
        fk_col: ForeignKeySchema,
    ):
        """Allocates worst-case buffers for a topology update phase."""
        # Example scenarios:
        # A -> B -> C
        # If B gets deleted. A.next and C.prev need to be updated
        # So 2 edits split across the next and prev arrays.
        #
        # A -> B -> C -> D
        # If B and C get deleted we still have only 2 edits.
        #
        # Unlinks have capacity cost of 1 per array.
        #
        # 0A>A -> 0B>A -> 0C>A | 1A>B -> 1B>B | 2A>C -> 2B>C
        # If 0B>A get's relinked to C then 0A.next becomes 0C
        # and 2A.next becomes 0B because new elements get attached to the head.
        # So at most 2 edits per update/relink.
        #
        # A -> (B) -> C
        # If B is a newly added node A.next and C.prev need to be updated.
        #
        # So total max capacity for all unlinks, updates, and additions combined is n_unlinks + 2*updates + n_additions.

        capacity = n_unlinks + 2 * n_relinks + n_additions + n_moves

        target_idx_dtype = fk_col.target_table.index_spec.dtype
        parent_idx_dtype = fk_col.parent_table.index_spec.dtype

        adj_conf = fk_col.adjacency_conf

        count_capacity = adj_conf.track_counts * capacity

        return cls(
            next_indices=np.empty(capacity, dtype=parent_idx_dtype),
            next_values=np.empty(capacity, dtype=parent_idx_dtype),
            prev_indices=np.empty(capacity, dtype=parent_idx_dtype),
            prev_values=np.empty(capacity, dtype=parent_idx_dtype),
            head_indices=np.empty(capacity, dtype=target_idx_dtype),
            head_values=np.empty(capacity, dtype=parent_idx_dtype),
            count_indices=np.empty(count_capacity, dtype=target_idx_dtype),
            count_deltas=np.empty(count_capacity, dtype=np.int64),
            nullify_indices=fk_nulls.astype(parent_idx_dtype),
        )

    def translate_in_place(
        self,
        source_oracle: RemapOracle,
        target_oracle: RemapOracle,
    ):
        """Translate patch buffers to final physical index space."""
        if self.next_cursor:
            self.next_indices = oracle_resolve_array(
                self.next_indices[: self.next_cursor], source_oracle, inplace=True
            )
            self.next_values = oracle_resolve_array(
                self.next_values[: self.next_cursor], source_oracle, inplace=True
            )
            valid = self.next_indices != source_oracle.missing_index_sentinel
            if not valid.all():
                self.next_indices = self.next_indices[valid]
                self.next_values = self.next_values[valid]
                self.next_cursor = len(self.next_indices)

        if self.prev_cursor:
            self.prev_indices = oracle_resolve_array(
                self.prev_indices[: self.prev_cursor], source_oracle, inplace=True
            )
            self.prev_values = oracle_resolve_array(
                self.prev_values[: self.prev_cursor], source_oracle, inplace=True
            )
            valid = self.prev_indices != source_oracle.missing_index_sentinel
            if not valid.all():
                self.prev_indices = self.prev_indices[valid]
                self.prev_values = self.prev_values[valid]
                self.prev_cursor = len(self.prev_indices)

        if self.head_cursor:
            self.head_indices = oracle_resolve_array(
                self.head_indices[: self.head_cursor], target_oracle, inplace=True
            )
            self.head_values = oracle_resolve_array(
                self.head_values[: self.head_cursor], source_oracle, inplace=True
            )
            valid = self.head_indices != target_oracle.missing_index_sentinel
            if not valid.all():
                self.head_indices = self.head_indices[valid]
                self.head_values = self.head_values[valid]
                self.head_cursor = len(self.head_indices)

        if self.count_cursor:
            self.count_indices = oracle_resolve_array(
                self.count_indices[: self.count_cursor], target_oracle, inplace=True
            )

            valid_mask = self.count_indices != target_oracle.missing_index_sentinel

            self.count_indices, self.count_deltas = (
                self.count_indices[valid_mask],
                self.count_deltas[valid_mask],
            )
            self.count_cursor = len(self.count_indices)

        if len(self.nullify_indices):
            resolved = oracle_resolve_array(
                self.nullify_indices[
                    self.nullify_indices != source_oracle.missing_index_sentinel
                ],
                source_oracle,
            )
            self.nullify_indices = resolved[
                resolved != source_oracle.missing_index_sentinel
            ]

    def apply(
        self,
        source_table: PackedArrayTable,
        target_table: PackedArrayTable,
        fk_col: ForeignKeySchema,
    ):
        """Applies sparse updates to the physical arrays."""
        if self.next_cursor > 0:
            source_table[fk_col.adj_next].view[
                self.next_indices[: self.next_cursor]
            ] = self.next_values[: self.next_cursor]

        if self.prev_cursor > 0:
            source_table[fk_col.adj_prev].view[
                self.prev_indices[: self.prev_cursor]
            ] = self.prev_values[: self.prev_cursor]

        if self.head_cursor > 0:
            target_table[fk_col.adj_head].view[
                self.head_indices[: self.head_cursor]
            ] = self.head_values[: self.head_cursor]

        if self.count_cursor > 0 and fk_col.adjacency_conf.track_counts:
            # Atomic add handles multiple children affecting the same parent count
            np.add.at(
                target_table[fk_col.adj_count].view,
                self.count_indices[: self.count_cursor],
                self.count_deltas[: self.count_cursor],
            )

        if len(self.nullify_indices) > 0:
            source_table[fk_col].view[self.nullify_indices] = (
                source_table.schema.index_spec.missing
            )


class RemapOracle(NamedTuple):
    """An oracle that answers where an index has moved during a bulk edit.

    Because swap-and-pop edits move row indices, references (like foreign keys)
    need to be remapped to their new physical locations. This oracle provides
    the mapping rules for a specific table during a transaction commit.

    Attributes:
        set_null_unlinks_sorted: Sorted array of indices that should be set to null.
        deletions_sorted: Sorted array of deleted row indices.
        moves_from: Sorted array of original row indices that were relocated.
        moves_to: The new physical indices corresponding to `moves_from`.
        moves_to_sorted: A sorted version of `moves_to` for fast collision checks.
        addition_destinations: Array of indices where new virtual rows were placed.
        virtual_indices_start: The index where virtual (newly added) rows begin.
        new_size: The total physical size of the table after the edit.
        missing_index_sentinel: The integer sentinel representing a missing link.
    """
    set_null_unlinks_sorted: np.ndarray
    deletions_sorted: np.ndarray
    moves_from: np.ndarray
    moves_to: np.ndarray
    moves_to_sorted: np.ndarray
    addition_destinations: np.ndarray
    virtual_indices_start: int
    new_size: int
    missing_index_sentinel: int


@nb.njit(inline="always")
def oracle_resolve(idx, oracle: RemapOracle) -> int:
    o = oracle
    missing_idx = o.missing_index_sentinel

    # Virtual IDs (New Additions)
    if idx >= o.virtual_indices_start:
        offset = idx - o.virtual_indices_start
        if offset < len(o.addition_destinations):
            return int(o.addition_destinations[offset])
        return missing_idx

    # Explicit Deletions
    if len(o.deletions_sorted) > 0:
        # Optimization: fast range check before binary search
        if o.deletions_sorted[0] <= idx <= o.deletions_sorted[-1]:
            i_del = np.searchsorted(o.deletions_sorted, idx)
            if i_del < len(o.deletions_sorted) and o.deletions_sorted[i_del] == idx:
                return missing_idx

    # Moves (Did I move?)
    if len(o.moves_from) > 0:
        if o.moves_from[0] <= idx <= o.moves_from[-1]:
            i_move = np.searchsorted(o.moves_from, idx)
            if i_move < len(o.moves_from) and o.moves_from[i_move] == idx:
                return int(o.moves_to[i_move])

    # Overwritten by Move (Did someone move here?)
    if len(o.moves_to_sorted) > 0:
        if o.moves_to_sorted[0] <= idx <= o.moves_to_sorted[-1]:
            i_over = np.searchsorted(o.moves_to_sorted, idx)
            if i_over < len(o.moves_to_sorted) and o.moves_to_sorted[i_over] == idx:
                return missing_idx

    # Overwritten by Addition (Did an addition land here?)
    if len(o.addition_destinations) > 0:
        if o.addition_destinations[0] <= idx <= o.addition_destinations[-1]:
            i_add = np.searchsorted(o.addition_destinations, idx)
            if (
                i_add < len(o.addition_destinations)
                and o.addition_destinations[i_add] == idx
            ):
                return missing_idx

    # Truncation (Array shrunk)
    if idx >= o.new_size:
        return missing_idx

    return idx


@nb.njit(cache=True, parallel=True)
def oracle_resolve_array(
    values: np.ndarray,
    oracle: RemapOracle,
    inplace: bool = False,
) -> np.ndarray:
    """Translate a list of indices through a RemapOracle.

    This resolves:
      - virtual indices
      - moved indices
      - overwritten indices
      - deletions

    Args:
        values: Array of indices referencing the oracle's table.
        oracle: RemapOracle for the referenced table.
        inplace: Preform the update in-place on the provided values array.

    Returns:
        New array with all indices translated to final physical indices
        or missing sentinel.
    """
    if not inplace:
        out = np.zeros_like(values)
    else:
        out = values

    missing = oracle.missing_index_sentinel

    for i in nb.prange(len(values)):
        v = values[i]
        if v == missing:
            out[i] = missing
        else:
            out[i] = oracle_resolve(v, oracle)

    return out


@dataclass(slots=True)
class DeletionTraceback:
    steps: list[TableSchema | ForeignKeySchema] = field(
        init=False, default_factory=list
    )

    def start(self, table: SupportsGetTableSchema):
        self.steps = [table.get_table_schema()]

    def new_stage(self, col: ForeignKeySchema):
        self.steps.append(col)

    def stage_finished(self):
        self.steps.pop()

    def __str__(self) -> str:
        str_steps = []
        for step in self.steps:
            if isinstance(step, TableSchema):
                str_steps.append(step.name)
            elif isinstance(step, ForeignKeySchema):
                str_steps.append(
                    f"{step.on_delete.name}->'{step.parent_table.name}'.'{step.name}'"
                )

        return "->".join(str_steps)


@dataclass(slots=True)
class TransactionContext:
    """A transaction context that's used to efficiently stage and bulk commit edits.

    Batches additions, updates, and deletions across all tables. When the context
    exits, it calculates a global bulk-edit plan, updates the adjacency lists
    safely across all foreign keys, and commits the changes in one pass.
    """

    db: PackedArrayDB

    additions: dict[str, tuple[list[np.ndarray], ...]] = field(
        init=False, default_factory=dict
    )
    """`TableName -> Tuple[ Column -> List[ArrayChunks] ]`"""

    deletions: defaultdict[str, set[int]] = field(
        init=False,
        default_factory=lambda: defaultdict(set),
    )
    """`TableName -> Set[ Rows ]`"""

    fk_set_null_unlinks: defaultdict[tuple[str, int], set[int]] = field(
        init=False, default_factory=lambda: defaultdict(set)
    )
    """FKs to unlink without deleting `Tuple[ TableName, ColID ] -> Set[ Rows ]`"""

    new_fk_claims: defaultdict[str, set[int]] = field(
        init=False,
        default_factory=lambda: defaultdict(set),
    )
    """`TableName -> Set[ Rows ]`"""

    updates: defaultdict[str, defaultdict[str, dict[int, Any]]] = field(
        init=False, default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    """Stores value updates: `TableName -> [ ColName -> [ RowIdx -> Value ] ]`"""

    fk_updates: defaultdict[str, defaultdict[str, dict[int, int]]] = field(
        init=False, default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    """Stores FK topology changes: `TableName -> [ ColName -> [ RowIdx -> NewTargetID ] ]`"""

    _virtual_counters: dict[str, int] = field(init=False)
    _virtual_starts: dict[str, int] = field(init=False)

    _deletion_traceback: DeletionTraceback = field(
        init=False, default_factory=DeletionTraceback
    )

    def __post_init__(self):
        self._virtual_starts = {t.schema.name: len(t) for t in self.db.tables}
        self._virtual_counters = self._virtual_starts.copy()

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.db._transaction_finished()
        if exc_type is None:
            self.commit()

    # --- Edits registration ---

    def register_updates_col_major(
        self, table: TableSchema, updates: NormalizedUpdatesColMajor
    ):
        """Registers bulk updates."""
        tbl_name = table.name
        deleted_set = self.deletions[tbl_name]
        virtual_start = self._virtual_starts[tbl_name]

        # Pre-validate all updates before applying partial changes
        for col_idx, indices, _ in updates:
            col_obj = table.cols[col_idx]
            col_name = col_obj.name

            # Check for overlap with new additions (Virtual IDs)
            if np.max(indices) >= virtual_start:
                raise UpdateVirtualRowException(
                    message=f"Virtual row(s) in update indices for '{tbl_name}'.'{col_name}':\n{indices[indices >= virtual_start]}"
                )

            # Check for overlap with deletions
            # Fast intersection check using sets
            idx_set = set(indices)
            if not deleted_set.isdisjoint(idx_set):
                invalid = deleted_set.intersection(idx_set)
                raise UpdateDeletedRowException(
                    message=f"Deleted rows in update indices for '{tbl_name}'.'{col_name}': {invalid}",
                    table_name=tbl_name,
                    col_schema=col_obj,
                    problematic_indices=invalid,
                )

            # Check for overlap with queued updates (Physical IDs)
            pending_col_updates = self.updates[tbl_name].get(col_name, {})
            for idx in indices:
                if idx < virtual_start and idx in pending_col_updates:
                    raise UpdateQueuedEditException(
                        message=f"Row {idx} in '{tbl_name}'.'{col_name}' already has a staged update.",
                        problematic_index=idx,
                        table_name=tbl_name,
                        col_schema=col_obj,
                    )

        tbl_obj = self.db.get_table(table)

        # Apply Updates
        for col_idx, indices, values in updates:
            col_schema = table.cols[col_idx]
            col_name = col_schema.name
            is_fk = isinstance(col_schema, ForeignKeySchema)
            true_updates = np.nonzero(values != tbl_obj[col_schema].view[indices])[0]

            for i in range(len(true_updates)):
                idx = int(indices[true_updates][i])
                val = values[true_updates][i]

                # Queue Physical Update
                self.updates[tbl_name][col_name][idx] = val

                if is_fk:
                    # Queue Topology Change
                    self.fk_updates[tbl_name][col_name][idx] = val

    def register_additions_col_major(
        self, table: TableSchema, values: NormalizedRecordsColMajor
    ) -> range:
        additions = self.additions.get(table.name)
        if additions is None:
            additions = tuple([] for _ in range(len(table.cols)))
            self.additions[table.name] = additions

        deleted_set = self.deletions[table.name]

        for i, col in enumerate(table.cols):
            if isinstance(col, ForeignKeySchema):
                if col.target_table.name in self.deletions:
                    targets = values[i]

                    if not deleted_set.isdisjoint(targets):
                        raise VoidReferenceException(
                            message=f"Received keys for '{col.name}' that target deleted rows"
                        )

        for i, col in enumerate(table.cols):
            additions[i].append(values[i])
            if isinstance(col, ForeignKeySchema):
                self.new_fk_claims[col.target_table.name].update(values[i])

        current_counter = self._virtual_counters[table.name]
        n_additions = len(values[0]) if values else 0
        new_counter = current_counter + n_additions
        new_virtual_ids = range(current_counter, new_counter)

        self._virtual_counters[table.name] = new_counter

        return new_virtual_ids

    def register_additions_row_major(
        self, table: TableSchema, values: tuple[NormalizedRecordRowMajor, ...]
    ) -> range:
        additions_col_maj = tuple(np.array(a) for a in zip(*values, strict=True))
        return self.register_additions_col_major(table, additions_col_maj)

    def register_deletions(self, table: TableSchema, indices: Iterable[int]):
        """Mark rows for deletion.

        Args:
            table: The target table to delete rows from.
            indices: The indices of the rows to mark for deletion.

        Raises:
            DeleteNewlyClaimedException: When deleting a row claimed by a new entry.
            DeleteClaimedByStrictFkException: When deleting a row that's claimed by a `FkOnDeleteStyle.RESTRICT` key.
        """
        expanded_deletions, expanded_fk_deletions = self.expand_and_validate_deletions(
            table, indices
        )

        for k, v in expanded_deletions.items():
            self.deletions[k].update(v)

        for k, v in expanded_fk_deletions.items():
            self.fk_set_null_unlinks[k].update(v)

    @overload
    def expand_and_validate_deletions(
        self,
        table: SupportsGetTableSchema,
        indices: Iterable[int],
        prev_expanded_deletions: None = None,
        prev_expanded_fk_deletions: None = None,
    ) -> tuple[dict[str, set[int]], dict[tuple[str, int], set[int]]]: ...

    @overload
    def expand_and_validate_deletions(
        self,
        table: SupportsGetTableSchema,
        indices: Iterable[int],
        prev_expanded_deletions: defaultdict[str, list[int]],
        prev_expanded_fk_deletions: defaultdict[tuple[str, int], list[int]],
    ) -> None: ...

    def expand_and_validate_deletions(
        self,
        table: SupportsGetTableSchema,
        indices: Iterable[int],
        prev_expanded_deletions: defaultdict[str, list[int]] | None = None,
        prev_expanded_fk_deletions: defaultdict[tuple[str, int], list[int]]
        | None = None,
    ) -> tuple[dict[str, set[int]], dict[tuple[str, int], set[int]]] | None:
        table = table.get_table_schema()

        if prev_expanded_deletions is None and prev_expanded_fk_deletions is None:
            self._deletion_traceback.start(table)

        expanded_deletions = prev_expanded_deletions or defaultdict(list)
        expanded_fk_deletions = prev_expanded_fk_deletions or defaultdict(list)

        if table.name in self.new_fk_claims:
            intersection = self.new_fk_claims[table.name].intersection(indices)
            if intersection:
                raise DeleteNewlyClaimedException(
                    message="IDs are claimed by newly staged foreign key entries",
                    table_name=table.name,
                    problematic_indices=intersection,
                    traceback=self._deletion_traceback,
                )

        # Handle foreign keys of other tables that are subscribed to this one.
        for subscriber in table.subscribers:
            subscriber_tbl_name = subscriber.parent_table.name
            subscriber_tbl = self.db.get_table(subscriber_tbl_name)

            connected_indices = subscriber_tbl[subscriber].get_referencing_indices(
                indices
            )

            flat_connected_indices = set().union(*connected_indices)

            if (
                fk_updates_t := self.fk_updates.get(subscriber_tbl_name, None)
            ) is not None and (
                fk_updates_c := fk_updates_t.get(subscriber.name)
            ) is not None:
                flat_connected_indices -= fk_updates_c.keys()

            new_to_delete = flat_connected_indices.difference(
                self.deletions[subscriber_tbl_name],
                expanded_deletions[subscriber_tbl_name],
            )
            new_to_delete.discard(subscriber_tbl.schema.index_spec.missing)

            if new_to_delete:
                match subscriber.on_delete:
                    case FksOnDeleteStyle.RESTRICT:
                        raise DeleteClaimedByStrictFkException(
                            message=f"Some IDs are referenced by keys of on_delete = RESTRICT column '{subscriber_tbl_name}'.'{subscriber.name}'",
                            table_name=subscriber_tbl_name,
                            problematic_indices=new_to_delete,
                            traceback=self._deletion_traceback,
                        )
                    case FksOnDeleteStyle.SET_NULL:
                        expanded_fk_deletions[
                            (
                                subscriber_tbl_name,
                                subscriber_tbl.column_ids[subscriber],
                            )
                        ].extend(new_to_delete)
                    case FksOnDeleteStyle.CASCADE:
                        self.expand_and_validate_deletions(
                            subscriber_tbl,
                            new_to_delete,
                            expanded_deletions,
                            expanded_fk_deletions,
                        )

        expanded_deletions[table.name].extend(indices)

        if prev_expanded_deletions is None and prev_expanded_fk_deletions is None:
            final_deletions = {k: set(v) for k, v in expanded_deletions.items()}

            final_fk_deletions = {
                k: set(v).difference(final_deletions[k[0]])
                for k, v in expanded_fk_deletions.items()
            }

            return final_deletions, final_fk_deletions

    def get_dirty_tables(self) -> set[str]:
        dirty_tables = (
            set(self.additions.keys())
            | set(self.deletions.keys())
            | set(self.updates.keys())
        )
        for t, _ in self.fk_set_null_unlinks:
            dirty_tables.add(t)

        return dirty_tables

    # --- Commit logic ---

    def _prepare_additions(self):
        self.additions = {
            tbl: tuple([np.concatenate(chunks)] for chunks in cols)
            for tbl, cols in self.additions.items()
        }

    def _create_oracle(self, table: str) -> RemapOracle:
        table_obj = self.db.get_table(table)

        idx_spec = table_obj.schema.index_spec
        idx_dtype = idx_spec.dtype

        additions = self.additions.get(table, None)
        n_additions = 0 if additions is None else len(additions[0][0])

        arr_deletions = (
            np.fromiter(self.deletions[table], dtype=idx_dtype)
            if table in self.deletions
            else np.empty(0, dtype=idx_dtype)
        )

        new_size, addition_dests, (moves_from, moves_to), arr_deletions = (
            plan_bulk_edit(len(table_obj), n_additions, arr_deletions)
        )

        arr_moves_from = np.asarray(moves_from, dtype=idx_dtype)[::-1]
        arr_moves_to = np.asarray(moves_to, dtype=idx_dtype)[::-1]
        arr_moves_to_sorted = np.sort(arr_moves_to)
        arr_addition_dests = np.asarray(addition_dests, dtype=idx_dtype)

        return RemapOracle(
            set_null_unlinks_sorted=np.empty(0),
            deletions_sorted=arr_deletions,
            moves_from=arr_moves_from,
            moves_to=arr_moves_to,
            moves_to_sorted=arr_moves_to_sorted,
            addition_destinations=arr_addition_dests,
            virtual_indices_start=self._virtual_starts[table],
            new_size=new_size,
            missing_index_sentinel=idx_spec.missing,
        )

    def _create_fk_column_oracle(
        self,
        col: ForeignKeySchema,
        table_oracles: dict[str, RemapOracle] | None = None,
    ) -> RemapOracle:
        table = col.parent_table.name
        table_obj = self.db.get_table(table)

        table_oracle = (
            table_oracles[table]
            if table_oracles is not None and table in table_oracles
            else self._create_oracle(col.parent_table.name)
        )

        fk_unlinks = self.fk_set_null_unlinks.get(
            (table, table_obj.column_ids[col.name]), None
        )
        if fk_unlinks:
            col_oracle = RemapOracle(
                np.sort(
                    np.fromiter(fk_unlinks, dtype=table_obj.schema.index_spec.dtype)
                ),
                *table_oracle[1:],
            )
            return col_oracle

        return table_oracle

    def _get_identity_oracle(self, table: str) -> RemapOracle:
        table_obj = self.db.get_table(table)
        return RemapOracle(
            set_null_unlinks_sorted=np.empty(0),
            deletions_sorted=np.empty(0),
            moves_from=np.empty(0),
            moves_to=np.empty(0),
            moves_to_sorted=np.empty(0),
            addition_destinations=np.empty(0),
            virtual_indices_start=len(table_obj),
            new_size=len(table_obj),
            missing_index_sentinel=table_obj.schema.index_spec.missing,
        )

    def _compute_topology_patches(
        self, dirty_table: str, oracles: dict[str, RemapOracle]
    ) -> list[TopologyScratchpad]:
        table_obj = self.db.get_table(dirty_table)
        patches = []

        fk_col_ids = table_obj.foreign_key_columns

        for fk_id in fk_col_ids:
            fk = cast(ForeignKeySchema, table_obj.schema.cols[fk_id])
            source_tbl = table_obj
            target_tbl = self.db.get_table(fk.target_table)

            missing_idx_src = source_tbl.schema.index_spec.missing
            missing_idx_tgt = target_tbl.schema.index_spec.missing

            source_oracle = self._create_fk_column_oracle(fk, oracles)

            target_oracle = (
                oracles[target_tbl.name]
                if target_tbl.name in oracles
                else self._get_identity_oracle(target_tbl.name)
            )

            update_map = self.fk_updates.get(dirty_table, {}).get(fk.name, {})

            rows_to_unlink = np.union1d(
                source_oracle.deletions_sorted,
                source_oracle.set_null_unlinks_sorted,
            ).astype(source_tbl.schema.index_spec.dtype)

            if update_map:
                updates_sorted = np.sort(
                    np.fromiter(
                        update_map.keys(),
                        dtype=source_tbl.schema.index_spec.dtype,
                    )
                )
                rows_to_unlink = np.union1d(rows_to_unlink, updates_sorted)

            # Identify moves for rows that are NOT being unlinked/updated.
            # If a row is in rows_to_unlink, it is leaving its current chain,
            # so we must NOT patch its neighbors to point to its new physical location.
            if len(source_oracle.moves_from) > 0 and len(rows_to_unlink) > 0:
                # Both arrays are sorted, so we can use isin efficiently
                keep_mask = ~np.isin(source_oracle.moves_from, rows_to_unlink)
                active_moves_from = source_oracle.moves_from[keep_mask]
                active_moves_to = source_oracle.moves_to[keep_mask]
            else:
                active_moves_from = source_oracle.moves_from
                active_moves_to = source_oracle.moves_to

            additions = (
                self.additions[dirty_table][fk_id][0]
                if dirty_table in self.additions
                else np.empty(0, dtype=fk.target_table.index_spec.dtype)
            )
            n_additions = len(additions)

            newly_claimed_indices, newly_claimed_counts = np.unique(
                additions[additions != missing_idx_tgt], return_counts=True
            )

            head_ignore_indices = newly_claimed_indices
            if len(target_oracle.deletions_sorted) > 0:
                head_ignore_indices = np.union1d(
                    head_ignore_indices, target_oracle.deletions_sorted
                )

            # Handle Updates claiming new heads
            upd_claims = None
            upd_counts = None
            if update_map:
                updated_targets = np.fromiter(
                    update_map.values(), dtype=fk.target_table.index_spec.dtype
                )
                updated_targets_resolved = oracle_resolve_array(
                    updated_targets, target_oracle
                )

                upd_claims, upd_counts = np.unique(
                    updated_targets_resolved[
                        updated_targets_resolved != missing_idx_tgt
                    ],
                    return_counts=True,
                )
                head_ignore_indices = np.union1d(head_ignore_indices, upd_claims)

            # Allocate Scratchpad
            # We only pass source_oracle.set_null_unlinks_sorted as indices to physically nullify.
            scratch_pad = TopologyScratchpad.allocate(
                len(rows_to_unlink),
                n_additions,
                len(update_map),
                len(active_moves_from),
                source_oracle.set_null_unlinks_sorted,
                fk,
            )

            sp_next_indices = scratch_pad.next_indices
            sp_next_values = scratch_pad.next_values

            sp_prev_indices = scratch_pad.prev_indices
            sp_prev_values = scratch_pad.prev_values

            sp_head_indices = scratch_pad.head_indices
            sp_head_values = scratch_pad.head_values

            sp_next_cursor = scratch_pad.next_cursor
            sp_prev_cursor = scratch_pad.prev_cursor
            sp_head_cursor = scratch_pad.head_cursor

            fk_arr = source_tbl[fk].view
            adj_next = source_tbl[fk.adj_next].view
            adj_prev = source_tbl[fk.adj_prev].view
            adj_head = target_tbl[fk.adj_head].view

            (sp_next_cursor, sp_prev_cursor, sp_head_cursor) = (
                scratch_pad.next_cursor,
                scratch_pad.prev_cursor,
                scratch_pad.head_cursor,
            ) = patch_unlinked_adjacency_and_heads(
                missing_val_src=missing_idx_src,
                missing_val_tgt=missing_idx_tgt,
                unlinked_indices=rows_to_unlink,
                head_ignore_indices=head_ignore_indices,
                parent_fk=fk_arr,
                adj_next=adj_next,
                adj_prev=adj_prev,
                target_adj_head=adj_head,
                out_next_idx=sp_next_indices,
                out_next_val=sp_next_values,
                n_cursor=sp_next_cursor,
                out_prev_idx=sp_prev_indices,
                out_prev_val=sp_prev_values,
                p_cursor=sp_prev_cursor,
                out_head_idx=sp_head_indices,
                out_head_val=sp_head_values,
                h_cursor=sp_head_cursor,
            )

            if len(active_moves_from):
                (sp_next_cursor, sp_prev_cursor, sp_head_cursor) = (
                    scratch_pad.next_cursor,
                    scratch_pad.prev_cursor,
                    scratch_pad.head_cursor,
                ) = patch_relocated_adjacency_and_heads(
                    missing_val_src=missing_idx_src,
                    missing_val_tgt=missing_idx_tgt,
                    head_ignore_indices=head_ignore_indices,
                    moves_from=active_moves_from,
                    moves_to=active_moves_to,
                    src_fk=fk_arr,
                    adj_next=adj_next,
                    adj_prev=adj_prev,
                    target_adj_head=adj_head,
                    out_next_idx=sp_next_indices,
                    out_next_val=sp_next_values,
                    n_cursor=sp_next_cursor,
                    out_prev_idx=sp_prev_indices,
                    out_prev_val=sp_prev_values,
                    p_cursor=sp_prev_cursor,
                    out_head_idx=sp_head_indices,
                    out_head_val=sp_head_values,
                    h_cursor=sp_head_cursor,
                )

            if len(scratch_pad.count_indices):
                cnt_idxs = fk_arr[rows_to_unlink]
                cnt_idxs = cnt_idxs[cnt_idxs != missing_idx_tgt]
                cnt_deltas = np.full(len(cnt_idxs), -1, dtype=np.int64)

                # Combine decrements (cnt_idxs), new additions (newly_claimed),
                # and updates/re-links (upd_claims).

                index_dtype = source_tbl.schema.index_spec.dtype
                indices_list = [
                    cnt_idxs.astype(index_dtype),
                    newly_claimed_indices.astype(index_dtype),
                ]
                deltas_list = [
                    cnt_deltas.astype(index_dtype),
                    newly_claimed_counts.astype(index_dtype),
                ]

                if upd_claims is not None:
                    indices_list.append(upd_claims)
                    deltas_list.append(upd_counts)

                scratch_pad.count_indices = np.concatenate(indices_list)
                scratch_pad.count_deltas = np.concatenate(deltas_list)

                scratch_pad.count_cursor = len(scratch_pad.count_indices)

            if len(update_map):
                u_rows = np.fromiter(
                    update_map.keys(), dtype=source_tbl.schema.index_spec.dtype
                )
                u_targets = np.fromiter(
                    update_map.values(), dtype=target_tbl.schema.index_spec.dtype
                )

                u_rows_res = oracle_resolve_array(u_rows, source_oracle)
                u_targets_res = oracle_resolve_array(u_targets, target_oracle)

                valid = u_rows_res != missing_idx_src
                if np.any(valid):
                    (sp_next_cursor, sp_prev_cursor, sp_head_cursor) = (
                        scratch_pad.next_cursor,
                        scratch_pad.prev_cursor,
                        scratch_pad.head_cursor,
                    ) = interleave_existing_rows(
                        row_indices=u_rows_res[valid],
                        targets=u_targets_res[valid],
                        current_heads=target_tbl[fk.adj_head].view,
                        adj_next=adj_next,
                        unlinked_indices=rows_to_unlink,
                        missing_val_src=missing_idx_src,
                        missing_val_tgt=missing_idx_tgt,
                        out_next_idx=sp_next_indices,
                        out_next_val=sp_next_values,
                        n_cursor=sp_next_cursor,
                        out_prev_idx=sp_prev_indices,
                        out_prev_val=sp_prev_values,
                        p_cursor=sp_prev_cursor,
                        out_head_idx=sp_head_indices,
                        out_head_val=sp_head_values,
                        h_cursor=sp_head_cursor,
                    )

            if n_additions and len(additions) > 0:
                targets = cast(np.ndarray, additions)

                curr_heads = target_tbl[fk.adj_head].view

                (_, _, sp_prev_cursor, sp_head_cursor) = (
                    new_nexts,
                    new_prevs,
                    scratch_pad.prev_cursor,
                    scratch_pad.head_cursor,
                ) = interleave_new_rows(
                    n_new=n_additions,
                    virt_start=source_oracle.virtual_indices_start,
                    missing_val_src=missing_idx_src,
                    missing_val_tgt=missing_idx_tgt,
                    targets=targets,
                    current_heads=curr_heads,
                    adj_next=adj_next,
                    unlinked_indices=rows_to_unlink,
                    out_prev_idx=sp_prev_indices,
                    out_prev_val=sp_prev_values,
                    p_cursor=sp_prev_cursor,
                    out_head_idx=sp_head_indices,
                    out_head_val=sp_head_values,
                    h_cursor=sp_head_cursor,
                )

                final_fk_values = oracle_resolve_array(targets, target_oracle)
                oracle_resolve_array(new_nexts, source_oracle, inplace=True)
                oracle_resolve_array(new_prevs, source_oracle, inplace=True)
                self._patch_addition_buffers(
                    dirty_table, fk, final_fk_values, new_nexts, new_prevs
                )

            scratch_pad.translate_in_place(source_oracle, target_oracle)
            patches.append(scratch_pad)

        return patches

    def _patch_addition_buffers(
        self,
        table_name: str,
        fk: ForeignKeySchema,
        new_fks: np.ndarray,
        new_nexts: np.ndarray,
        new_prevs: np.ndarray,
    ):
        cols = list(self.additions[table_name])
        idx_fk = self.db.get_table(table_name).column_ids[fk]
        idx_next = self.db.get_table(table_name).column_ids[fk.adj_next]
        idx_prev = self.db.get_table(table_name).column_ids[fk.adj_prev]
        cols[idx_fk][0] = new_fks
        cols[idx_next][0] = new_nexts
        cols[idx_prev][0] = new_prevs
        self.additions[table_name] = tuple(cols)

    def commit(self):
        self._prepare_additions()
        dirty_tables = self.get_dirty_tables()

        oracles = {tbl: self._create_oracle(tbl) for tbl in dirty_tables}

        tbl_patches = {
            tbl: self._compute_topology_patches(tbl, oracles) for tbl in dirty_tables
        }

        # Materialization
        for tbl_name in dirty_tables:
            o = oracles[tbl_name]
            src_tbl = self.db.get_table(tbl_name)

            src_tbl._len = o.new_size

            additions = self.additions.get(tbl_name)
            for i, col in enumerate(src_tbl.schema.cols):
                col_obj = src_tbl[col]
                data = col_obj.view
                if len(o.moves_from) > 0:
                    data[o.moves_to] = data[o.moves_from]

                if additions:
                    if o.new_size > len(data):
                        src_tbl.arrays[i].ensure_size(o.new_size, virtual_shrink=True)
                        data = col_obj.view
                    data[o.addition_destinations] = additions[i][0]

                pending_updates = self.updates.get(tbl_name, {}).get(col.name)

                if pending_updates:
                    rows = np.fromiter(
                        pending_updates.keys(),
                        dtype=src_tbl.schema.index_spec.dtype,
                    )

                    if isinstance(col, ForeignKeySchema):
                        target_oracle = oracles.get(col.target_table.name)
                        vals = np.fromiter(
                            pending_updates.values(),
                            dtype=col.target_table.index_spec.dtype,
                        )
                        if target_oracle:
                            vals = oracle_resolve_array(vals, target_oracle)
                    else:
                        vals = np.fromiter(
                            pending_updates.values(),
                            dtype=cast(DataColSchema, col).dtype,
                        )

                    dest_rows = oracle_resolve_array(rows, o)
                    valid = dest_rows != o.missing_index_sentinel

                    if np.any(valid):
                        data[dest_rows[valid]] = vals[valid]

                if o.new_size < len(data):
                    src_tbl.arrays[i].ensure_size(o.new_size, virtual_shrink=True)

        # Apply patches
        for tbl_name, patches in tbl_patches.items():
            src_tbl = self.db.get_table(tbl_name)
            fk_cols = [
                c for c in src_tbl.schema.cols if isinstance(c, ForeignKeySchema)
            ]

            for patch, fk in zip(patches, fk_cols, strict=True):
                tgt_tbl = self.db.get_table(fk.target_table)
                patch.apply(src_tbl, tgt_tbl, fk)

        # Fix FKs that got severed by relocations
        for tgt_tbl_name in dirty_tables:
            tgt_oracle = oracles.get(tgt_tbl_name, None)

            if tgt_oracle and len(tgt_oracle.moves_from):
                tgt_tbl = self.db.get_table(tgt_tbl_name)
                for sub in tgt_tbl.schema.subscribers:
                    src_tbl = self.db.get_table(sub.parent_table)
                    src_col_arr = src_tbl[sub].view
                    src_tbl_name = sub.parent_table.name

                    src_oracle = oracles[src_tbl_name]

                    fix_keys_broken_by_moved_targets(
                        src_missing=src_oracle.missing_index_sentinel,
                        moves_to=tgt_oracle.moves_to,
                        head=tgt_tbl[sub.adj_head].view,
                        src_fk=src_col_arr,
                        adj_next=src_tbl[sub.adj_next].view,
                        adj_prev=src_tbl[sub.adj_prev].view,
                    )


@dataclass(kw_only=True)
class TransactionContextException(Exception):
    """Base exception that forces keyword-only metadata."""

    message: str

    def __post_init__(self):
        # This ensures the message is passed to the underlying
        # Exception logic for proper traceback printing.
        super().__init__(self.message)


@dataclass(kw_only=True)
class UpdateDeletedRowException(TransactionContextException):
    """Raised when trying to update a deleted row."""

    table_name: str
    col_schema: ColSchemaLike
    problematic_indices: set[int]

    ...


@dataclass(kw_only=True)
class UpdateQueuedEditException(TransactionContextException):
    """Raised when an update targets a row that has already been edited in the current transaction."""

    problematic_index: int
    table_name: str
    col_schema: ColSchemaLike

    ...


@dataclass(kw_only=True)
class UpdateVirtualRowException(TransactionContextException):
    """Raised when an update targets a virtual row."""

    ...


@dataclass(kw_only=True)
class VoidReferenceException(TransactionContextException):
    """Raised when a foreign key references a row marked for deletion."""

    ...


@dataclass(kw_only=True)
class DeleteClaimedException(TransactionContextException):
    """Raised when trying to mark a claimed row for deletion."""

    table_name: str
    """The table that the deletions target."""

    problematic_indices: set[int]
    traceback: DeletionTraceback

    def __str__(self) -> str:
        header = f"{self.__class__.__name__} in '{self.table_name}'"
        details = f"Indices: {self.problematic_indices}"
        path = f"Constraint Path: {self.traceback}"
        return f"{self.message.strip()}\n\n{header}\n{details}\n{path}"


@dataclass(kw_only=True)
class DeleteNewlyClaimedException(DeleteClaimedException):
    """Raised when trying to mark a row for deletion that's claimed by a new FK entry."""

    ...


@dataclass(kw_only=True)
class DeleteClaimedByStrictFkException(DeleteClaimedException):
    """Raised when trying to mark a row for deletion that's claimed by a `on_delete = RESTRICT` FK."""

    ...
