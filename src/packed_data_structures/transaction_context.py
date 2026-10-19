from __future__ import annotations
from packed_data_structures.schemas.injected_logic_columns import PmFkTypeIdColSchema
from typing import Literal
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from packed_data_structures.edit_helpers import (
    interleave_existing_rows,
    patch_relocated_adjacency_and_heads,
    patch_unlinked_adjacency_and_heads,
    interleave_new_rows,
    interleave_new_rows_non_contiguous,
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
    PolymorphicForeignKeySchema,
)
from packed_data_structures.table import NormalizedUpdatesColMajor, PackedArrayTable
from packed_data_structures.remap_oracle import RemapOracle, oracle_resolve_array
from packed_data_structures.trackers import (
    BaseTracker,
    SingleTracker,
    RangeTracker,
    ArrayTracker,
)

if TYPE_CHECKING:
    from packed_data_structures.database import PackedArrayDB


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

    # Metadata for applying the patch
    target_table_schema: TableSchema | None = None
    fk_col_schema: ColSchemaLike | None = None
    adj_next_schema: ColSchemaLike | None = None
    adj_prev_schema: ColSchemaLike | None = None
    adj_head_schema: ColSchemaLike | None = None
    adj_count_schema: ColSchemaLike | None = None
    track_counts: bool = False

    @classmethod
    def allocate(
        cls,
        n_unlinks: int,
        n_additions: int,
        n_relinks: int,
        n_moves: int,
        fk_nulls: np.ndarray,
        target_idx_dtype: type[np.integer[Any]],
        parent_idx_dtype: type[np.integer[Any]],
        track_counts: bool,
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

        count_capacity = track_counts * capacity

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
            track_counts=track_counts,
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
    ):
        """Applies sparse updates to the physical arrays."""
        if self.next_cursor > 0:
            assert self.adj_next_schema is not None
            source_table[self.adj_next_schema].view[
                self.next_indices[: self.next_cursor]
            ] = self.next_values[: self.next_cursor]

        if self.prev_cursor > 0:
            assert self.adj_prev_schema is not None
            source_table[self.adj_prev_schema].view[
                self.prev_indices[: self.prev_cursor]
            ] = self.prev_values[: self.prev_cursor]

        if self.head_cursor > 0:
            assert self.adj_head_schema is not None
            target_table[self.adj_head_schema].view[
                self.head_indices[: self.head_cursor]
            ] = self.head_values[: self.head_cursor]

        if self.count_cursor > 0 and self.track_counts:
            assert self.adj_count_schema is not None
            # Atomic add handles multiple children affecting the same parent count
            np.add.at(
                target_table[self.adj_count_schema].view,
                self.count_indices[: self.count_cursor],
                self.count_deltas[: self.count_cursor],
            )

        if len(self.nullify_indices) > 0:
            assert self.fk_col_schema is not None
            source_table[self.fk_col_schema].view[self.nullify_indices] = ()


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
    pm_fk_target_updates: defaultdict[str, defaultdict[str, dict[int, int]]] = field(
        init=False, default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    """Stores PM FK target changes: `TableName -> [ ColName -> [ RowIdx -> NewTargetID ] ]`"""
    pm_fk_type_id_updates: defaultdict[str, defaultdict[str, dict[int, int]]] = field(
        init=False, default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    """Stores PM FK target table (type_id) changes: `TableName -> [ ColName -> [ RowIdx -> NewTargetTypeID ] ]`"""

    _staged_counters: dict[str, int] = field(init=False)
    _staged_starts: dict[str, int] = field(init=False)

    _deletion_traceback: DeletionTraceback = field(
        init=False, default_factory=DeletionTraceback
    )

    _trackers: defaultdict[str, list[BaseTracker]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self):
        self._staged_starts = {t.schema.name: len(t) for t in self.db.tables}
        self._staged_counters = self._staged_starts.copy()

    def __enter__(self) -> TransactionContext:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.db._transaction_finished()
        if exc_type is None:
            self.commit()

    # --- Edits registration & tracking ---

    @overload
    def create_tracker(
        self,
        table: str | SupportsGetTableSchema,
        ids: int,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ) -> SingleTracker: ...

    @overload
    def create_tracker(
        self,
        table: str | SupportsGetTableSchema,
        ids: range,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ) -> RangeTracker: ...

    @overload
    def create_tracker(
        self,
        table: str | SupportsGetTableSchema,
        ids: Sequence[int] | np.ndarray,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ) -> ArrayTracker: ...

    def create_tracker(
        self,
        table: str | SupportsGetTableSchema,
        ids: int | range | Sequence[int] | np.ndarray,
        storage_method: Literal["auto", "hard", "soft"] = "hard",
    ) -> BaseTracker:
        """Creates an explicit tracker to resolve physical IDs after a commit.

        When working within a transaction, adding new entries or deleting existing
        entries will cause the physical array layout to shift (via swap-and-pop).
        Trackers safely resolve either staged IDs (newly added entries) or
        original IDs (existing entries) to their final physical locations.

        Args:
            table: The table or schema the tracked IDs belong to.
            ids: The ID or sequence of IDs to track. Can be a single integer,
                a range object, a list/sequence of integers, or a numpy array.
            storage_method: The strategy for managing the tracker's internal data:
                - `"hard"`: Creates a clean, isolated copy of the mapped IDs.
                  Safest option to prevent accidental memory leaks from the oracle.
                - `"soft"`: Creates a direct view into the oracle's buffers. More
                  efficient, but holding the tracker prevents oracle memory cleanup.
                - `"auto"`: Dynamically chooses between `"hard"` and `"soft"` based
                  on memory footprint heuristics (e.g. % of array viewed). Defaults
                  to `"hard"`.

        Returns:
            BaseTracker: A specific tracker subclass (`SingleTracker`, `RangeTracker`,
                or `ArrayTracker`) that can be queried or reified post-commit.

        Raises:
            TypeError: If the provided `ids` type is not supported.
        """
        schema = self.db.get_table_schema(table)
        idx_dtype = schema.index_spec.dtype

        if isinstance(ids, int):
            tracker = SingleTracker(ids, idx_dtype, storage_method)
        elif isinstance(ids, range):
            tracker = RangeTracker(ids, idx_dtype, storage_method)
        elif isinstance(ids, (np.ndarray, Sequence)):
            tracker = ArrayTracker(ids, idx_dtype, storage_method)
        else:
            raise TypeError(f"Unsupported ids type: {type(ids)}")

        self._trackers[schema.name].append(tracker)
        return tracker

    def register_updates_col_major(
        self, table: TableSchema, updates: NormalizedUpdatesColMajor
    ):
        """Registers bulk updates."""
        tbl_name = table.name
        deleted_set = self.deletions[tbl_name]
        staging_start = self._staged_starts[tbl_name]

        # Pre-validate all updates before applying partial changes
        for col_idx, indices, _ in updates:
            col_obj = table.cols[col_idx]
            col_name = col_obj.name

            # Check for overlap with new additions (Staged IDs)
            if np.max(indices) >= staging_start:
                raise UpdateStagedRowException(
                    message=f"Staged row(s) in update indices for '{tbl_name}'.'{col_name}':\n{indices[indices >= staging_start]}"
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
                if idx < staging_start and idx in pending_col_updates:
                    raise UpdateQueuedEditException(
                        message=f"Row {idx} in '{tbl_name}'.'{col_name}' already has a staged update.",
                        problematic_index=idx,
                        table_name=tbl_name,
                        col_schema=col_obj,
                    )

        tbl_obj = self.db.get_table(table)

        # Stage all physical updates
        for col_idx, indices, values in updates:
            col_schema = table.cols[col_idx]
            col_name = col_schema.name
            is_fk = isinstance(col_schema, ForeignKeySchema)
            is_pm_fk = isinstance(col_schema, PolymorphicForeignKeySchema)
            is_pm_fk_type_id = isinstance(col_schema, PmFkTypeIdColSchema)
            true_updates = np.nonzero(values != tbl_obj[col_schema].view[indices])[0]
            true_indices = indices[true_updates]

            if is_pm_fk:
                current_type_ids = self.db.get_table(table)[
                    col_schema.type_id_col
                ].view[true_indices]

            if is_pm_fk_type_id:
                current_pm_fk_targets = self.db.get_table(table)[
                    col_schema.parent_col
                ].view[true_indices]

            if is_pm_fk or is_pm_fk_type_id:
                pm_fk_col = col_schema if is_pm_fk else col_schema.parent_col
                pm_fk_type_id_updates = self.pm_fk_type_id_updates[tbl_name][
                    pm_fk_col.name
                ]
                pm_fk_target_updates = self.pm_fk_target_updates[tbl_name][
                    pm_fk_col.name
                ]

            for i in range(len(true_updates)):
                idx = int(indices[true_updates][i])
                val = values[true_updates][i]

                # Queue physical update
                self.updates[tbl_name][col_name][idx] = val

                # Queue topology change
                if is_fk:
                    self.fk_updates[tbl_name][col_name][idx] = val
                elif is_pm_fk:
                    pm_fk_target_updates[idx] = val
                    if idx not in pm_fk_type_id_updates:
                        pm_fk_type_id_updates[idx] = current_type_ids[i]
                elif is_pm_fk_type_id:
                    pm_fk_type_id_updates[idx] = val
                    if idx not in pm_fk_target_updates:
                        pm_fk_target_updates[idx] = current_pm_fk_targets[i]

    def register_additions_col_major(
        self, table: TableSchema, new_records: tuple[np.ndarray, ...]
    ) -> range:
        additions = self.additions.get(table.name)
        if additions is None:
            additions = tuple([] for _ in range(len(table.cols)))
            self.additions[table.name] = additions

        for i, col in enumerate(table.cols):
            if isinstance(col, ForeignKeySchema):
                if col.target_table.name not in self.deletions:
                    continue
                targets = new_records[i]
                target_deletions = self.deletions[col.target_table.name]

                if not target_deletions.isdisjoint(targets):
                    raise VoidReferenceException(
                        message=f"Received keys for '{col.name}' that target deleted rows"
                    )
            elif isinstance(col, PolymorphicForeignKeySchema):
                type_id_data = new_records[table.col_ids[col.type_id_col]]
                targets = new_records[i]

                for type_id, target_table in enumerate(col.target_tables):
                    if target_table.name not in self.deletions:
                        continue

                    mask = type_id_data == type_id
                    relevant_targets = targets[mask]

                    if not self.deletions[target_table.name].isdisjoint(
                        relevant_targets
                    ):
                        raise VoidReferenceException(
                            message=f"Received keys for '{col.name}' that target deleted rows in {target_table.name}"
                        )

        for i, col in enumerate(table.cols):
            additions[i].append(new_records[i])
            if isinstance(col, ForeignKeySchema):
                self.new_fk_claims[col.target_table.name].update(new_records[i])
            elif isinstance(col, PolymorphicForeignKeySchema):
                type_id_data = new_records[table.col_ids[col.type_id_col]]
                targets = new_records[i]
                for type_id, target_table in enumerate(col.target_tables):
                    relevant_targets = targets[type_id_data == type_id]
                    self.new_fk_claims[target_table.name].update(relevant_targets)

        current_counter = self._staged_counters[table.name]
        n_additions = len(new_records[0]) if new_records else 0
        new_counter = current_counter + n_additions

        self._staged_counters[table.name] = new_counter

        # Return staged indices
        return range(current_counter, new_counter)

    def register_additions_row_major(
        self, table: TableSchema, new_records: tuple[tuple[Any, ...], ...]
    ) -> range:
        if not new_records:
            additions_col_maj = tuple(
                np.empty(0, dtype=self.db.get_table(table.name).arrays[i].dtype)
                for i in range(len(table.cols))
            )
        else:
            additions_col_maj = tuple(
                np.array(a, dtype=self.db.get_table(table.name).arrays[i].dtype)
                for i, a in enumerate(zip(*new_records, strict=True))
            )
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

            if isinstance(subscriber, PolymorphicForeignKeySchema):
                type_id = subscriber.type_id_mapping[table]
                connected_indices = subscriber_tbl[subscriber].get_referencing_indices(
                    type_id, indices
                )
            else:
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
            set_null_unlinks_sorted=np.empty(0, dtype=idx_dtype),
            deletions_sorted=arr_deletions,
            moves_from=arr_moves_from,
            moves_to=arr_moves_to,
            moves_to_sorted=arr_moves_to_sorted,
            addition_destinations=arr_addition_dests,
            staged_indices_start=self._staged_starts[table],
            new_size=new_size,
            missing_index_sentinel=idx_spec.missing,
        )

    def _create_fk_column_oracle(
        self,
        col: ForeignKeySchema | PolymorphicForeignKeySchema,
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
        idx_dtype = table_obj.schema.index_spec.dtype
        return RemapOracle(
            set_null_unlinks_sorted=np.empty(0, dtype=idx_dtype),
            deletions_sorted=np.empty(0, dtype=idx_dtype),
            moves_from=np.empty(0, dtype=idx_dtype),
            moves_to=np.empty(0, dtype=idx_dtype),
            moves_to_sorted=np.empty(0, dtype=idx_dtype),
            addition_destinations=np.empty(0, dtype=idx_dtype),
            staged_indices_start=len(table_obj),
            new_size=len(table_obj),
            missing_index_sentinel=table_obj.schema.index_spec.missing,
        )

    def _process_topology_patches_for_subgraph(
        self,
        dirty_table: str,
        source_tbl: PackedArrayTable[np.integer[Any]],
        target_tbl: PackedArrayTable[np.integer[Any]],
        fk: ForeignKeySchema | PolymorphicForeignKeySchema,
        source_oracle: RemapOracle,
        target_oracle: RemapOracle,
        unlinks_to_process: np.ndarray,
        global_unlinks: np.ndarray,
        relink_map: dict[int, int],
        additions: np.ndarray,
        additions_staged_indices: np.ndarray | None,
        active_moves_from: np.ndarray,
        active_moves_to: np.ndarray,
        fk_nulls: np.ndarray,
        fk_arr: np.ndarray,
        adj_next: np.ndarray,
        adj_prev: np.ndarray,
        target_adj_head: np.ndarray,
        target_adj_head_schema: ColSchemaLike,
        target_adj_count_schema: ColSchemaLike | None,
        missing_idx_src: int,
        missing_idx_tgt: int,
    ) -> tuple[TopologyScratchpad, np.ndarray, np.ndarray, np.ndarray]:
        n_additions = len(additions)

        newly_claimed_indices, newly_claimed_counts = np.unique(
            additions[additions != missing_idx_tgt], return_counts=True
        )

        head_ignore_indices = newly_claimed_indices
        if len(target_oracle.deletions_sorted) > 0:
            head_ignore_indices = np.union1d(
                head_ignore_indices, target_oracle.deletions_sorted
            )

        upd_claims = None
        upd_counts = None
        if relink_map:
            updated_targets = np.fromiter(
                relink_map.values(), dtype=target_tbl.schema.index_spec.dtype
            )
            updated_targets_resolved = oracle_resolve_array(
                updated_targets, target_oracle
            )

            upd_claims, upd_counts = np.unique(
                updated_targets_resolved[updated_targets_resolved != missing_idx_tgt],
                return_counts=True,
            )
            head_ignore_indices = np.union1d(head_ignore_indices, upd_claims)

        scratch_pad = TopologyScratchpad.allocate(
            len(unlinks_to_process),
            n_additions,
            len(relink_map),
            len(active_moves_from),
            fk_nulls,
            target_tbl.schema.index_spec.dtype,
            source_tbl.schema.index_spec.dtype,
            fk.adjacency_conf.track_counts,
        )

        scratch_pad.target_table_schema = target_tbl.schema
        scratch_pad.fk_col_schema = fk
        scratch_pad.adj_next_schema = fk.adj_next
        scratch_pad.adj_prev_schema = fk.adj_prev
        scratch_pad.adj_head_schema = target_adj_head_schema
        scratch_pad.adj_count_schema = target_adj_count_schema

        sp_next_indices = scratch_pad.next_indices
        sp_next_values = scratch_pad.next_values

        sp_prev_indices = scratch_pad.prev_indices
        sp_prev_values = scratch_pad.prev_values

        sp_head_indices = scratch_pad.head_indices
        sp_head_values = scratch_pad.head_values

        (sp_next_cursor, sp_prev_cursor, sp_head_cursor) = (
            scratch_pad.next_cursor,
            scratch_pad.prev_cursor,
            scratch_pad.head_cursor,
        ) = patch_unlinked_adjacency_and_heads(
            missing_val_src=missing_idx_src,
            missing_val_tgt=missing_idx_tgt,
            unlinked_indices=unlinks_to_process,
            head_ignore_indices=head_ignore_indices,
            parent_fk=fk_arr,
            adj_next=adj_next,
            adj_prev=adj_prev,
            target_adj_head=target_adj_head,
            out_next_idx=sp_next_indices,
            out_next_val=sp_next_values,
            n_cursor=scratch_pad.next_cursor,
            out_prev_idx=sp_prev_indices,
            out_prev_val=sp_prev_values,
            p_cursor=scratch_pad.prev_cursor,
            out_head_idx=sp_head_indices,
            out_head_val=sp_head_values,
            h_cursor=scratch_pad.head_cursor,
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
                target_adj_head=target_adj_head,
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
            cnt_idxs = fk_arr[unlinks_to_process]
            cnt_idxs = cnt_idxs[cnt_idxs != missing_idx_tgt]
            cnt_deltas = np.full(len(cnt_idxs), -1, dtype=np.int64)

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
                assert upd_counts is not None
                deltas_list.append(upd_counts)

            scratch_pad.count_indices = np.concatenate(indices_list)
            scratch_pad.count_deltas = np.concatenate(deltas_list)

            scratch_pad.count_cursor = len(scratch_pad.count_indices)

        if len(relink_map):
            u_rows = np.fromiter(
                relink_map.keys(), dtype=source_tbl.schema.index_spec.dtype
            )
            u_targets = np.fromiter(
                relink_map.values(), dtype=target_tbl.schema.index_spec.dtype
            )

            # Resolve to check validity (to drop deleted rows), but do not pass the resolved
            # indices to `interleave_existing_rows`. The `interleave_existing_rows` function
            # writes patches directly to `TopologyScratchpad`, which are subject to `oracle_resolve`
            # natively inside `TopologyScratchpad.translate_in_place()`.
            # If we wrote resolved indices, they would be translated twice and become broken.
            u_rows_res = oracle_resolve_array(u_rows, source_oracle)

            valid = u_rows_res != missing_idx_src
            if np.any(valid):
                (sp_next_cursor, sp_prev_cursor, sp_head_cursor) = (
                    scratch_pad.next_cursor,
                    scratch_pad.prev_cursor,
                    scratch_pad.head_cursor,
                ) = interleave_existing_rows(
                    row_indices=u_rows[valid],
                    targets=u_targets[valid],
                    current_heads=target_adj_head,
                    adj_next=adj_next,
                    unlinked_indices=global_unlinks,
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
            targets = additions

            curr_heads = target_adj_head

            if additions_staged_indices is None:
                (_, _, sp_prev_cursor, sp_head_cursor) = (
                    new_nexts,
                    new_prevs,
                    scratch_pad.prev_cursor,
                    scratch_pad.head_cursor,
                ) = interleave_new_rows(
                    n_new=n_additions,
                    staged_start=source_oracle.staged_indices_start,
                    missing_val_src=missing_idx_src,
                    missing_val_tgt=missing_idx_tgt,
                    targets=targets,
                    current_heads=curr_heads,
                    adj_next=adj_next,
                    unlinked_indices=global_unlinks,
                    out_prev_idx=sp_prev_indices,
                    out_prev_val=sp_prev_values,
                    p_cursor=sp_prev_cursor,
                    out_head_idx=sp_head_indices,
                    out_head_val=sp_head_values,
                    h_cursor=sp_head_cursor,
                )
            else:
                (_, _, sp_prev_cursor, sp_head_cursor) = (
                    new_nexts,
                    new_prevs,
                    scratch_pad.prev_cursor,
                    scratch_pad.head_cursor,
                ) = interleave_new_rows_non_contiguous(
                    n_new=n_additions,
                    staged_indices=additions_staged_indices,
                    missing_val_src=missing_idx_src,
                    missing_val_tgt=missing_idx_tgt,
                    targets=targets,
                    current_heads=curr_heads,
                    adj_next=adj_next,
                    unlinked_indices=global_unlinks,
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
        else:
            final_fk_values = np.empty(0, dtype=target_tbl.schema.index_spec.dtype)
            new_nexts = np.empty(0, dtype=source_tbl.schema.index_spec.dtype)
            new_prevs = np.empty(0, dtype=source_tbl.schema.index_spec.dtype)

        scratch_pad.translate_in_place(source_oracle, target_oracle)
        return scratch_pad, final_fk_values, new_nexts, new_prevs

    def _compute_topology_patches(
        self, dirty_table: str, oracles: dict[str, RemapOracle]
    ) -> list[TopologyScratchpad]:
        table_obj = self.db.get_table(dirty_table)
        patches = []

        fk_col_ids = table_obj.foreign_key_columns

        for fk_id in fk_col_ids:
            col_schema = table_obj.schema.cols[fk_id]
            source_tbl = table_obj

            if isinstance(col_schema, ForeignKeySchema):
                fk = col_schema
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

                global_unlinks = np.union1d(
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
                    global_unlinks = np.union1d(global_unlinks, updates_sorted)

                if len(source_oracle.moves_from) > 0 and len(global_unlinks) > 0:
                    keep_mask = ~np.isin(source_oracle.moves_from, global_unlinks)
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

                fk_arr = source_tbl[fk].view
                adj_next = source_tbl[fk.adj_next].view
                adj_prev = source_tbl[fk.adj_prev].view
                adj_head = target_tbl[fk.adj_head].view
                fk_nulls = source_oracle.set_null_unlinks_sorted

                patch, new_fks, new_nexts, new_prevs = (
                    self._process_topology_patches_for_subgraph(
                        dirty_table=dirty_table,
                        source_tbl=source_tbl,
                        target_tbl=target_tbl,
                        fk=fk,
                        source_oracle=source_oracle,
                        target_oracle=target_oracle,
                        unlinks_to_process=global_unlinks,
                        global_unlinks=global_unlinks,
                        relink_map=update_map,
                        additions=additions,
                        additions_staged_indices=None,
                        active_moves_from=active_moves_from,
                        active_moves_to=active_moves_to,
                        fk_nulls=fk_nulls,
                        fk_arr=fk_arr,
                        adj_next=adj_next,
                        adj_prev=adj_prev,
                        target_adj_head=adj_head,
                        target_adj_head_schema=fk.adj_head,
                        target_adj_count_schema=fk.adj_count
                        if fk.adjacency_conf.track_counts
                        else None,
                        missing_idx_src=missing_idx_src,
                        missing_idx_tgt=missing_idx_tgt,
                    )
                )
                if len(new_fks) > 0:
                    self._patch_addition_buffers(
                        dirty_table, fk, new_fks, new_nexts, new_prevs
                    )
                patches.append(patch)

            elif isinstance(col_schema, PolymorphicForeignKeySchema):
                pm_fk = col_schema
                source_oracle = self._create_fk_column_oracle(pm_fk, oracles)

                # Fetch updates maps
                target_updates = self.pm_fk_target_updates.get(dirty_table, {}).get(
                    pm_fk.name, {}
                )
                type_updates = self.pm_fk_type_id_updates.get(dirty_table, {}).get(
                    pm_fk.name, {}
                )

                # Physical old types
                old_types = source_tbl[pm_fk.type_id_col].view

                # Reconstruct new types for updated rows
                new_types = old_types.copy()
                for idx, val in type_updates.items():
                    new_types[idx] = val

                global_unlinks = np.union1d(
                    source_oracle.deletions_sorted,
                    source_oracle.set_null_unlinks_sorted,
                ).astype(source_tbl.schema.index_spec.dtype)

                if target_updates:
                    updates_sorted = np.sort(
                        np.fromiter(
                            target_updates.keys(),
                            dtype=source_tbl.schema.index_spec.dtype,
                        )
                    )
                    global_unlinks = np.union1d(global_unlinks, updates_sorted)

                if len(source_oracle.moves_from) > 0 and len(global_unlinks) > 0:
                    keep_mask = ~np.isin(source_oracle.moves_from, global_unlinks)
                    global_moves_from = source_oracle.moves_from[keep_mask]
                    global_moves_to = source_oracle.moves_to[keep_mask]
                else:
                    global_moves_from = source_oracle.moves_from
                    global_moves_to = source_oracle.moves_to

                global_additions = (
                    self.additions[dirty_table][fk_id][0]
                    if dirty_table in self.additions
                    else np.empty(
                        0, dtype=pm_fk.keys_idx_spec.dtype
                    )  # PM-FK target dtype
                )
                type_id_col_id = table_obj.column_ids[pm_fk.type_id_col]
                global_addition_types = (
                    self.additions[dirty_table][type_id_col_id][0]
                    if dirty_table in self.additions
                    else np.empty(0, dtype=pm_fk.type_id_dtype)
                )

                fk_arr = source_tbl[pm_fk].view
                adj_next = source_tbl[pm_fk.adj_next].view
                adj_prev = source_tbl[pm_fk.adj_prev].view

                missing_idx_src = source_tbl.schema.index_spec.missing

                for target_type_id, tgt_schema in enumerate(pm_fk.target_tables):
                    target_tbl = self.db.get_table(tgt_schema.name)
                    missing_idx_tgt = (
                        pm_fk.keys_idx_spec.missing
                    )  # PM-FK uses shared index spec!

                    target_oracle = (
                        oracles[target_tbl.name]
                        if target_tbl.name in oracles
                        else self._get_identity_oracle(target_tbl.name)
                    )

                    # Filter Unlinks using OLD type
                    src_idx_dtype = source_tbl.schema.index_spec.dtype
                    deletions_sorted = source_oracle.deletions_sorted.astype(
                        src_idx_dtype
                    )
                    mask_del = old_types[deletions_sorted] == target_type_id
                    target_deletions = deletions_sorted[mask_del]

                    unlinks_sorted = source_oracle.set_null_unlinks_sorted.astype(
                        src_idx_dtype
                    )
                    mask_unl = old_types[unlinks_sorted] == target_type_id
                    target_unlinks = unlinks_sorted[mask_unl]

                    unlinks_to_process = np.union1d(
                        target_deletions, target_unlinks
                    ).astype(src_idx_dtype)

                    # Add updates where old_type == target_type_id to unlinks_to_process
                    if target_updates:
                        mask_upd = old_types[updates_sorted] == target_type_id
                        target_updates_unlinks = updates_sorted[mask_upd]
                        unlinks_to_process = np.union1d(
                            unlinks_to_process, target_updates_unlinks
                        )

                    # Filter Relinks using NEW type
                    relink_map = {}
                    for idx, target_val in target_updates.items():
                        if new_types[idx] == target_type_id:
                            relink_map[idx] = target_val

                    # Filter Additions using NEW type
                    if len(global_additions) > 0:
                        mask_add = global_addition_types == target_type_id
                        additions = global_additions[mask_add]
                    else:
                        mask_add = np.empty(0, dtype=bool)
                        additions = global_additions

                    # Filter Moves using OLD type (since moves don't change type)
                    if len(global_moves_from) > 0:
                        mask_mov = old_types[global_moves_from] == target_type_id
                        active_moves_from = global_moves_from[mask_mov]
                        active_moves_to = global_moves_to[mask_mov]
                    else:
                        active_moves_from = global_moves_from
                        active_moves_to = global_moves_to

                    target_adj_head = target_tbl[
                        pm_fk.adj_head_columns[target_type_id]
                    ].view

                    patch, new_fks, new_nexts, new_prevs = (
                        self._process_topology_patches_for_subgraph(
                            dirty_table=dirty_table,
                            source_tbl=source_tbl,
                            target_tbl=target_tbl,
                            fk=pm_fk,
                            source_oracle=source_oracle,
                            target_oracle=target_oracle,
                            unlinks_to_process=unlinks_to_process,
                            global_unlinks=global_unlinks,
                            relink_map=relink_map,
                            additions=additions,
                            additions_staged_indices=(
                                source_oracle.staged_indices_start
                                + np.where(mask_add)[0]
                            ).astype(source_tbl.schema.index_spec.dtype),
                            active_moves_from=active_moves_from,
                            active_moves_to=active_moves_to,
                            fk_nulls=target_unlinks,
                            fk_arr=fk_arr,
                            adj_next=adj_next,
                            adj_prev=adj_prev,
                            target_adj_head=target_adj_head,
                            target_adj_head_schema=pm_fk.adj_head_columns[
                                target_type_id
                            ],
                            target_adj_count_schema=pm_fk.adj_count_columns[
                                target_type_id
                            ]
                            if pm_fk.adjacency_conf.track_counts
                            else None,
                            missing_idx_src=missing_idx_src,
                            missing_idx_tgt=missing_idx_tgt,
                        )
                    )

                    if len(global_additions) > 0 and len(new_fks) > 0:
                        self._patch_pm_addition_buffers(
                            dirty_table, pm_fk, new_fks, new_nexts, new_prevs, mask_add
                        )

                    patches.append(patch)

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

    def _patch_pm_addition_buffers(
        self,
        table_name: str,
        fk: PolymorphicForeignKeySchema,
        new_fks: np.ndarray,
        new_nexts: np.ndarray,
        new_prevs: np.ndarray,
        mask: np.ndarray,
    ):
        cols = list(self.additions[table_name])
        idx_fk = self.db.get_table(table_name).column_ids[fk]
        idx_next = self.db.get_table(table_name).column_ids[fk.adj_next]
        idx_prev = self.db.get_table(table_name).column_ids[fk.adj_prev]

        # Modify the arrays in place via mask
        cols[idx_fk][0][mask] = new_fks
        cols[idx_next][0][mask] = new_nexts
        cols[idx_prev][0][mask] = new_prevs

    def commit(self):
        self._prepare_additions()
        dirty_tables = self.get_dirty_tables()

        oracles = {tbl: self._create_oracle(tbl) for tbl in dirty_tables}

        # Fill trackers for dirty tables
        for tbl_name, oracle in oracles.items():
            if tbl_name in self._trackers:
                for tracker in self._trackers[tbl_name]:
                    tracker._fill(oracle)

        # Fill trackers for non-dirty tables (identity mapping)
        for tbl_name, trackers in self._trackers.items():
            if tbl_name not in dirty_tables:
                oracle = self._get_identity_oracle(tbl_name)
                for tracker in trackers:
                    tracker._fill(oracle)

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
                        src_tbl.arrays[i].ensure_size(o.new_size, shrink_size=True)
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
                    elif isinstance(col, PolymorphicForeignKeySchema):
                        vals = np.fromiter(
                            pending_updates.values(),
                            dtype=col.keys_idx_spec.dtype,
                        )

                        type_id_pending = self.updates.get(tbl_name, {}).get(
                            col.type_id_col.name, {}
                        )

                        # Apply target oracle resolution row by row
                        for idx_in_update, row_idx in enumerate(pending_updates.keys()):
                            type_id = type_id_pending.get(
                                row_idx, src_tbl[col.type_id_col].view[row_idx]
                            )
                            target_tbl_name = col.target_tables[type_id].name
                            target_oracle = oracles.get(target_tbl_name)
                            if target_oracle:
                                vals[idx_in_update] = oracle_resolve_array(
                                    np.array(
                                        [vals[idx_in_update]],
                                        dtype=col.keys_idx_spec.dtype,
                                    ),
                                    target_oracle,
                                )[0]
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
                    src_tbl.arrays[i].ensure_size(o.new_size, shrink_size=True)

        # Apply patches
        for tbl_name, patches in tbl_patches.items():
            src_tbl = self.db.get_table(tbl_name)

            for patch in patches:
                assert patch.target_table_schema is not None
                tgt_tbl = self.db.get_table(patch.target_table_schema)
                patch.apply(src_tbl, tgt_tbl)

        # Fix FKs that got severed by relocations
        for tgt_tbl_name in dirty_tables:
            tgt_oracle = oracles.get(tgt_tbl_name, None)

            if tgt_oracle and len(tgt_oracle.moves_from):
                tgt_tbl = self.db.get_table(tgt_tbl_name)
                for sub in tgt_tbl.schema.subscribers:
                    src_tbl = self.db.get_table(sub.parent_table)
                    src_col_arr = src_tbl[sub].view

                    if isinstance(sub, PolymorphicForeignKeySchema):
                        type_id = sub.type_id_mapping[tgt_tbl.schema]
                        adj_head_name = sub.adj_head_columns[type_id]
                    else:
                        adj_head_name = sub.adj_head

                    fix_keys_broken_by_moved_targets(
                        src_missing=src_tbl.schema.index_spec.missing,
                        moves_to=tgt_oracle.moves_to,
                        head=tgt_tbl[adj_head_name].view,
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
class UpdateStagedRowException(TransactionContextException):
    """Raised when an update targets a staged row."""

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
