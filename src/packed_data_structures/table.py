from __future__ import annotations
from packed_data_structures.schemas.object_col import ObjectColSchema
from packed_data_structures.schema_accessors.object_col_accessor import (
    ObjectColSchemaAccessor,
)
from packed_data_structures import DataColSchema
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from collections.abc import Iterable, Sequence
from collections.abc import Iterator

import numpy as np


if TYPE_CHECKING:
    from packed_data_structures.transaction_context import TransactionContext
    from packed_data_structures.database import PackedArrayDB

from packed_data_structures.arrays.dirty_tracking import (
    DirtyTimestampProvider,
    ProvidesDirtyTimestamp,
)

from packed_data_structures.arrays import PackedArray
from packed_data_structures.schemas import (
    ColSchemaLike,
    ForeignKeySchema,
    SupportsGetTableSchema,
    TableSchema,
)

from .schema_accessors import SchemaAccessor, ForeignKeySchemaAccessor

type RecordsColMajor = (
    dict[ColSchemaLike, Sequence[Any]] | Sequence[tuple[ColSchemaLike, Sequence[Any]]]
)
type NormalizedRecordsColMajor = tuple[np.ndarray, ...]

type RecordRowMajor = (
    dict[ColSchemaLike, Any] | dict[str, Any] | Sequence[tuple[ColSchemaLike, Any]]
    # | NormalizedRecordRowMajor
)
type NormalizedRecordRowMajor = tuple[Any, ...]

type UpdatesColMajor = (
    dict[ColSchemaLike | str, dict[int, Any] | tuple[Sequence[int], Sequence[Any]]]
    | Sequence[
        tuple[ColSchemaLike | str, dict[int, Any] | tuple[Sequence[int], Sequence[Any]]]
    ]
)
type NormalizedUpdatesColMajor = tuple[tuple[int, np.ndarray, np.ndarray], ...]


@dataclass(slots=True)
class PackedArrayTable[T_idx: np.integer[Any]](
    ProvidesDirtyTimestamp, SupportsGetTableSchema
):
    """A tabular data structure backed by columnar PackedArrays.

    This is the primary user-facing class for interacting with the database.
    It provides a schema-driven API, where users interact with rows and columns
    using the singleton schema objects (e.g., `DataColSchema`) instead of string keys.

    Attributes:
        db: The parent database instance containing this table.
        table_id: The internal ID of this table within the database.
        schema: The TableSchema defining the columns and index spec.
        column_ids: A mapping from schema objects (or string names) to array indices.
        arrays: A tuple of the underlying PackedArray buffers.
        foreign_key_columns: Indices of columns that represent foreign keys.
        name: The string identifier of the table.
    """

    db: PackedArrayDB
    table_id: int
    schema: TableSchema[T_idx]

    column_ids: dict[ColSchemaLike | str, int] = field(init=False, default_factory=dict)
    arrays: tuple[PackedArray, ...] = field(init=False)
    foreign_key_columns: tuple[int, ...] = field(init=False)

    name: str = field(init=False)

    _len: int = field(init=False, default=0)
    _default_record: tuple[Any, ...] = field(init=False)

    _dirty_sources_cache: tuple[DirtyTimestampProvider, ...] | None = field(
        init=False, default=None
    )

    def __post_init__(self):
        self.name = self.schema.name
        self.column_ids = {
            **self.schema.col_ids,
            **{k.name: v for k, v in self.schema.col_ids.items()},
        }
        self.arrays = self.schema.init_arrays()

        self.foreign_key_columns = tuple(
            i
            for i, col in enumerate(self.schema.cols)
            if isinstance(col, ForeignKeySchema)
        )
        self._default_record = tuple(
            np.dtype(a.dtype).type(a.empty_fill) for a in self.arrays
        )

    def __len__(self) -> int:
        return self._len

    @overload
    def __getitem__(
        self, key: str
    ) -> SchemaAccessor[ColSchemaLike[np.generic], np.generic]: ...

    @overload
    def __getitem__[T: np.integer[Any], T_counts: np.integer[Any]](
        self, key: ForeignKeySchema[T, T_idx, T_counts]
    ) -> ForeignKeySchemaAccessor[T, T_idx, T_counts]: ...

    @overload
    def __getitem__[T: np.generic, *T_shape](
        self,
        key: ObjectColSchema[T],
    ) -> ObjectColSchemaAccessor[T]: ...

    @overload
    def __getitem__[T: np.generic, *T_shape](
        self,
        key: DataColSchema[T, *T_shape],  # ty:ignore[invalid-type-arguments]
    ) -> SchemaAccessor[DataColSchema[T, *T_shape], T, *T_shape]: ...  # ty:ignore[invalid-type-arguments]

    @overload
    def __getitem__[T: np.generic](
        self, key: ColSchemaLike[T]
    ) -> SchemaAccessor[ColSchemaLike[T], T]: ...

    @overload
    def __getitem__(self, key: int) -> tuple[Any, ...]: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[tuple[Any, ...]]: ...

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"Row index out of range: {key}")

            return tuple(arr.view[key] for arr in self.arrays)
        elif isinstance(key, slice):
            views = tuple(arr.view[key] for arr in self.arrays)
            return tuple(zip(*views, strict=True))

        col_id = self.column_ids.get(key)
        if col_id is None:
            raise KeyError(
                f"Table '{self.schema.name}' does not contain column '{key.name}'"
            )

        match key:
            case ForeignKeySchema():
                return ForeignKeySchemaAccessor(self, key, col_id)
            case ObjectColSchema():
                return ObjectColSchemaAccessor(self, key, col_id)

        return SchemaAccessor(self, key, col_id)

    def get_table_schema(self) -> TableSchema:
        """Get the schema definition of this table.

        Returns:
            The underlying TableSchema.
        """
        return self.schema

    def normalize_records_row_major(
        self,
        *records: dict[ColSchemaLike, Any]
        | dict[str, Any]
        | Sequence[tuple[ColSchemaLike, Any]],
    ) -> Iterator[NormalizedRecordRowMajor]:
        default_record = list(self._default_record)
        for r in records:
            r_norm = default_record.copy()
            for k, v in r.items() if isinstance(r, dict) else r:
                assert isinstance(k, str | ColSchemaLike)

                col_id = self.column_ids.get(k)
                if col_id is None:
                    raise KeyError(
                        f"Table '{self.schema.name}' does not contain the included column '{k.name if isinstance(k, ColSchemaLike) else k}'"
                    )
                r_norm[col_id] = v

            yield tuple(r_norm)

    @overload
    def normalize_records_col_major(
        self, records: dict[ColSchemaLike, Sequence[Any]]
    ) -> NormalizedRecordsColMajor: ...

    @overload
    def normalize_records_col_major(
        self, records: Sequence[tuple[ColSchemaLike, Sequence[Any]]]
    ) -> NormalizedRecordsColMajor: ...

    def normalize_records_col_major(
        self,
        records: dict[ColSchemaLike, Sequence[Any]]
        | Sequence[tuple[ColSchemaLike, Sequence[Any]]],
    ) -> NormalizedRecordsColMajor:
        last_key: ColSchemaLike | None = None
        last_key = cast(
            ColSchemaLike | None, last_key
        )  # Fixes TY not seeing the walrus updates
        if isinstance(records, dict):
            try:
                records = cast(dict[ColSchemaLike, Sequence[Any]], records)
                records_intermediate = tuple(
                    (self.column_ids[last_key := k], v) for k, v in records.items()
                )
            except KeyError:
                assert last_key is not None and isinstance(
                    last_key, str | ColSchemaLike
                )
                raise KeyError(
                    f"Table '{self.schema.name}' does not contain the included column '{last_key if isinstance(last_key, str) else last_key.name}'"
                ) from None
        else:
            try:
                records_intermediate = tuple(
                    (self.column_ids[last_key := k], v) for k, v in records
                )
            except KeyError:
                assert last_key is not None and isinstance(
                    last_key, str | ColSchemaLike
                )
                raise KeyError(
                    f"Table '{self.schema.name}' does not contain the included column '{last_key if isinstance(last_key, str) else last_key.name}'"
                ) from None

        n_records = max(len(values) for _, values in records_intermediate)

        if any(len(values) != n_records for _, values in records_intermediate):
            raise ValueError(
                "Could not normalize columnar records data. Received misaligned columns."
            )

        included_keys = set(k for k, _ in records_intermediate)
        normalized_records: list[None | np.ndarray] = [None] * len(self.arrays)

        for col_id, values in records_intermediate:
            if isinstance(values, np.ndarray):
                normalized_records[col_id] = values
            else:
                normalized_records[col_id] = np.array(
                    values, dtype=self.arrays[col_id].dtype
                )

        for col_id in included_keys.symmetric_difference(self.column_ids.values()):
            normalized_records[col_id] = np.full(
                (n_records, *self.arrays[col_id].element_shape),
                self.arrays[col_id].empty_fill,
                dtype=self.arrays[col_id].dtype,
            )

        return cast(tuple[np.ndarray, ...], tuple(normalized_records))

    def normalize_updates_col_major(
        self, updates: UpdatesColMajor
    ) -> NormalizedUpdatesColMajor:
        """Normalizes various update formats into a standard column-major list of arrays."""
        if isinstance(updates, dict):
            iterator = updates.items()
        else:
            iterator = updates

        normalized = []

        for key, data in iterator:
            assert isinstance(key, str | ColSchemaLike)
            col_id = self.column_ids.get(key)
            if col_id is None:
                raise KeyError(f"Column '{key}' not found in table '{self.name}'")

            dtype = self.arrays[col_id].dtype

            if isinstance(data, dict):
                # {RowIdx: Value} map
                indices = np.fromiter(data.keys(), dtype=int, count=len(data))
                values = np.fromiter(data.values(), dtype=dtype, count=len(data))
            elif isinstance(data, tuple) and len(data) == 2:
                # (Indices, Values) tuple
                indices = np.asanyarray(data[0], dtype=int)
                values = np.asanyarray(data[1], dtype=dtype)
            else:
                raise ValueError(f"Invalid update data format for column {key}")

            if len(indices) != len(values):
                raise ValueError(
                    f"Update length mismatch for column {key}: {len(indices)} indices vs {len(values)} values"
                )

            normalized.append((col_id, indices, values))

        return tuple(normalized)

    def _check_transaction_context(self) -> TransactionContext:
        ctx = self.db._transaction_ctx
        if ctx is None:
            raise RuntimeError(
                "Could not perform edit operation. No active transaction context"
            )
        return ctx

    def update_entries(
        self,
        updates: UpdatesColMajor,
        shape: Literal["col_major", "row_major"] = "col_major",
    ) -> None:
        """Updates existing entries.

        Supports dictionary maps `{row_idx: val}` or tuple arrays `(indices, values)`.

        Args:
            updates: The update data structure.
            shape: structure of the update data (currently only 'col_major' supported for batching).
        """
        ctx = self._check_transaction_context()

        # Currently only implementing col_major path as it's the primary bulk interface
        if shape == "col_major":
            normalized = self.normalize_updates_col_major(updates)
            ctx.register_updates_col_major(self.schema, normalized)
        else:
            raise NotImplementedError("row_major updates not yet implemented")

    @overload
    def add_entries(
        self,
        records: Sequence[RecordRowMajor],
        records_shape: Literal["row_major"],
    ) -> range: ...

    @overload
    def add_entries(
        self,
        records: RecordsColMajor,
        records_shape: Literal["col_major"],
    ) -> range: ...

    def add_entries(
        self,
        records: Sequence[RecordRowMajor] | RecordsColMajor,
        records_shape: Literal["row_major", "col_major"],
    ) -> range:
        ctx = self._check_transaction_context()

        if records_shape == "row_major":
            records = cast(Sequence[RecordRowMajor], records)
            normalized_records = tuple(self.normalize_records_row_major(*records))
            return ctx.register_additions_row_major(self.schema, normalized_records)
        else:
            records = cast(RecordsColMajor, records)
            normalized_records = self.normalize_records_col_major(records)
            return ctx.register_additions_col_major(self.schema, normalized_records)

    def add_entry(
        self,
        entry_record: RecordRowMajor,
    ) -> int:
        return self.add_entries((entry_record,), "row_major")[0]

    def del_entries(
        self, indices: Iterable[int] | np.ndarray[Any, np.dtype[np.bool_]]
    ) -> None:
        """Delete rows.

        If any foreign keys are severed as a result of the deletion.
        Those rows are also deleted.

        Args:
            indices: The indices of the rows to delete

        Raises:
            IndexError: If any index doesn't satisfy:
                -1 < i < table_size
        """
        ctx = self._check_transaction_context()

        # Handle boolean mask
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            # Fast conversion to indices
            indices = np.nonzero(indices)[0]
        else:
            indices = tuple(indices)

        max_index = len(self) - 1
        for index in indices:
            if index < 0 or index > max_index:
                raise IndexError(
                    f"Could not perform delete operation. Encountered invalid index: {index}"
                )

        ctx.register_deletions(self.schema, indices)

    def del_entry(self, index: int):
        """Delete row.

        Args:
            index: The index of the row to delete

        Raises:
            IndexError: If the index doesn't satisfy:
                -1 < i < table_size
        """
        self.del_entries((index,))

    # --- dirty tracking ---
    def _collect_dirty_sources(self) -> tuple[DirtyTimestampProvider, ...]:
        return self.arrays
