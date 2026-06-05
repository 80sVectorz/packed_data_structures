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
    dtype: np.dtype[T]
    missing: int
    max_value: int

    @classmethod
    def from_dtype(cls, dtype_like: DTypeLike) -> IndexSpec:
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


@dataclass
class ColSchemaLike[T: np.generic]:
    name: str
    parent_table: TableSchema = field(init=False)

    def set_parent(self, parent: TableSchema):
        self.parent_table = parent

    def init_array(self) -> PackedArray[T]: ...

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class DataColSchema[T: np.generic](ColSchemaLike[T]):
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


@dataclass
class ForeignKeySchema[T: np.generic](ColSchemaLike[T]):
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
    @abstractmethod
    def get_table_schema(self) -> TableSchema: ...


@dataclass(slots=True)
class TableSchema(SupportsGetTableSchema):
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
        if new_subscriber not in self.subscribers:
            self.subscribers.append(new_subscriber)

    def init_arrays(
        self,
    ) -> tuple[PackedArray, ...]:
        self._finalized = True
        return tuple(col.init_array() for col in self.cols)

    def register_new_column(self, col: ColSchemaLike):
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
