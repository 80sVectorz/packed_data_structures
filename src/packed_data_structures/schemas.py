from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike

from packed_data_structures.packed_array import PackedArray


@dataclass(frozen=True, slots=True)
class IndexSpec[T: np.generic]:
    """Specification for integer indices used to address rows.

    Defines the underlying numpy data type, the sentinel value indicating
    a missing or null link, and the maximum valid row index.

    Attributes:
        dtype: The numpy integer data type for the index.
        missing: The sentinel integer value representing a missing index.
        max_value: The maximum valid integer value for an index.
    """
    dtype: np.dtype[T]
    missing: int
    max_value: int

    @classmethod
    def from_dtype(cls, dtype_like: DTypeLike) -> IndexSpec:
        """Create an IndexSpec from a numpy data type.

        By standard convention, the maximum representable value of the given
        integer type is reserved as the missing/sentinel value.

        Args:
            dtype_like: A numpy integer data type or equivalent.

        Returns:
            A new IndexSpec instance.

        Raises:
            TypeError: If the provided dtype is not an integer type.
        """
        dtype = np.dtype(dtype_like)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(f"Index dtype must be integer, got {dtype}")

        info = np.iinfo(dtype)  # type: ignore
        # Standard convention: Max value is the sentinel
        return cls(
            dtype=cast(np.dtype[T], dtype), missing=info.max, max_value=info.max - 1
        )

    def new_array(self, size: int) -> np.ndarray[Any, np.dtype[T]]:
        """Helper to allocate raw numpy arrays with correct initialization."""
        arr = np.empty(size, dtype=self.dtype)
        arr[:] = self.missing
        return arr


@dataclass(eq=False)
class ColSchemaLike[T: np.generic]:
    """Base class for all column schemas.

    Column schemas are treated as object-identity singletons (`id(self)`)
    when used as dictionary keys or accessors. They define a single column
    within a TableSchema.

    Attributes:
        name: The string identifier for the column.
        parent_table: The TableSchema instance this column belongs to.
    """
    name: str
    parent_table: TableSchema = field(init=False)

    def set_parent(self, parent: TableSchema):
        """Bind this column to a parent TableSchema.

        Args:
            parent: The TableSchema that will own this column.
        """
        self.parent_table = parent

    def init_array(self) -> PackedArray[T]: ...

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass(eq=False)
class DataColSchema[T: np.generic](ColSchemaLike[T]):
    """Schema for a standard data column containing raw values.

    Defines the data type, default values, and shape of elements within
    the column. Maps directly to a PackedArray buffer at runtime.

    Attributes:
        name: The string identifier for the column.
        dtype: The numpy data type of the column's elements.
        default: The default value used to fill empty or newly allocated slots.
        shape: The shape of individual elements. An empty tuple indicates scalar values.
    """
    name: str
    dtype: DTypeLike
    default: Any | tuple[Any, ...] = 0
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)
        if len(self.shape) != 0 and not isinstance(self.default, tuple):
            self.default = tuple(np.full(self.shape, self.default))

    def init_array(
        self,
    ) -> PackedArray[T]:
        return PackedArray(
            self.parent_table.pre_allocate,
            self.dtype,
            self.default,
            element_shape=self.shape,
        )

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class AdjacencyListConf:
    """Config for adjacency list structure.

    Attributes:
        track_counts: If a counts column should be made.
        counts_dtype: Dtype of the element count column.
            If no dtype is provided the index dtype of the fk is used.
    """

    track_counts: bool = False
    counts_dtype: None | DTypeLike = None
    parent_fk: ForeignKeySchema = field(init=False)


class FksOnDeleteStyle(Enum):
    """Various ForeignKeySchema on referenced row delete behaviors."""

    CASCADE = auto()
    """Cascade the deletion by also deleting the FK row"""
    RESTRICT = auto()
    """Block the deletion and raise an exception"""
    SET_NULL = auto()
    """Set FK field to missing index value"""


@dataclass(eq=False)
class ForeignKeySchema[T: np.generic](ColSchemaLike[T]):
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
    adjacency_conf: AdjacencyListConf = field(default_factory=AdjacencyListConf)

    adj_next: DataColSchema = field(init=False)
    adj_prev: DataColSchema = field(init=False)
    adj_head: DataColSchema = field(init=False)
    adj_count: DataColSchema = field(init=False)

    def __post_init__(self):
        self.adjacency_conf.parent_fk = self
        self.target_table.subscribe(self)

    def set_parent(self, parent: TableSchema):
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
            self.adj_count = DataColSchema(
                f"_adj_count_{full_name}",
                self.adjacency_conf.counts_dtype or parent.index_spec.dtype,
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


class SupportsGetTableSchema(ABC):
    """Interface for objects that can provide a TableSchema."""

    @abstractmethod
    def get_table_schema(self) -> TableSchema:
        """Get the underlying TableSchema.

        Returns:
            The TableSchema instance.
        """
        ...


@dataclass(slots=True)
class TableSchema(SupportsGetTableSchema):
    """Schema definition for a flat, column-oriented table.

    A TableSchema aggregates multiple `ColSchemaLike` definitions and
    dictates how the `PackedArrayTable` initializes its raw buffers.

    Attributes:
        name: The string identifier for the table.
        index_spec: The specification defining the table's index type and capacity.
        cols: The list of column schemas defining the table structure.
        pre_allocate: The initial element capacity to allocate for the table's arrays.
    """
    name: str
    index_spec: IndexSpec
    cols: list[ColSchemaLike[Any]]
    pre_allocate: int = 0

    subscribers: list[ForeignKeySchema] = field(init=False, default_factory=list)
    col_ids: dict[ColSchemaLike, int] = field(init=False, default_factory=dict)
    _finalized: bool = field(init=False, default=False)

    def __post_init__(self):
        for col in self.cols:
            col.set_parent(self)

        for i, col in enumerate(self.cols):
            self.col_ids[col] = i

    def subscribe(self, new_subscriber: ForeignKeySchema):
        """Register a foreign key that targets this table.

        Args:
            new_subscriber: The foreign key schema pointing to this table.
        """
        if new_subscriber not in self.subscribers:
            self.subscribers.append(new_subscriber)

    def init_arrays(
        self,
    ) -> tuple[PackedArray, ...]:
        self._finalized = True
        return tuple(col.init_array() for col in self.cols)

    def register_new_column(self, col: ColSchemaLike):
        """Dynamically add a new column to the table schema.

        This is primarily used by overlay features and foreign keys to inject hidden
        management columns prior to initialization.

        Args:
            col: The column schema to add.

        Raises:
            RuntimeError: If the schema has already been initialized.
        """
        if self._finalized:
            raise RuntimeError(
                f"Cannot register column '{col.name}' to table '{self.name}': "
                "Schema is already finalized/initialized"
            )

        self.col_ids[col] = len(self.cols)

        self.cols.append(col)
        col.set_parent(self)

    def get_table_schema(self) -> TableSchema:
        return self
