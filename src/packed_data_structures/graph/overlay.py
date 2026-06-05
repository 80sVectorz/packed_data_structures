from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Self, Literal, overload

import numpy as np

from packed_data_structures.schemas import (
    AdjacencyListConf,
    IndexSpec,
    ForeignKeySchema,
    SupportsGetTableSchema,
    TableSchema,
    ColSchemaLike,
)
from packed_data_structures.overlays.base import DbOverlay
from packed_data_structures.overlays.registry import SchemaRegistry
from packed_data_structures.overlays.hooks import TransactionHook
from packed_data_structures.database import PackedArrayDB


class GraphFeature:
    """Base for components attached to graph layers.

    Encourages declarative schema definition via class attributes.
    """

    def on_attach(self, layer: BaseGraphLayer):
        """Runs when the feature is added to a Graph Layer."""
        pass

    def on_schema(self, registry: SchemaRegistry, layer_name: str):
        """Runs when the Graph Overlay's register_schema method is called."""
        pass

    def create_hook(self) -> TransactionHook | None:
        return None


class BaseGraphLayer(SupportsGetTableSchema):
    """Base class for Graph Layers."""

    name: str
    overlay: GraphOverlay
    schema: TableSchema
    features: list[GraphFeature]

    def __init__(self, name: str, overlay: GraphOverlay):
        self.name = name
        self.overlay = overlay
        self.features = []
        # Schema is initialized by subclasses

    def add_feature(self, feat: GraphFeature) -> Self:
        self.features.append(feat)
        feat.on_attach(self)
        return self

    def get_feature[T: GraphFeature](self, feature_type: type[T]) -> T:
        """Retrieves the first attached feature of the specified type."""
        for feat in self.features:
            if isinstance(feat, feature_type):
                return feat
        raise KeyError(
            f"Feature of type '{feature_type.__name__}' not found in layer '{self.name}'"
        )

    @overload
    def add_entry(self, data: dict[ColSchemaLike, Any], **kwargs) -> int: ...
    @overload
    def add_entry(self, **kwargs) -> int: ...

    def add_entry(self, data: dict[ColSchemaLike, Any] | None = None, **kwargs) -> int:
        """Add a row to the layer using Typed Keys (Schema Objects) or String Keys."""
        record = {}
        if data:
            record.update(data)
        record.update(kwargs)

        if not self.overlay._db:
            raise RuntimeError("DB not initialized")

        return self.overlay._db.get_table(self.name).add_entry(record)[0]

    def get_table_schema(self) -> TableSchema:
        return self.schema


class NodeLayer(BaseGraphLayer):
    def __init__(self, name: str, overlay: GraphOverlay):
        super().__init__(name, overlay)
        # Immediate Schema Instantiation
        spec = IndexSpec.from_dtype(overlay.index_dtype)
        self.schema = TableSchema(name, spec, [])


class EdgeLayer(BaseGraphLayer):
    src: ForeignKeySchema
    tgt: ForeignKeySchema

    def __init__(
        self, name: str, overlay: GraphOverlay, source: NodeLayer, target: NodeLayer
    ):
        super().__init__(name, overlay)
        spec = IndexSpec.from_dtype(overlay.index_dtype)
        self.schema = TableSchema(name, spec, [])

        # Immediate FK Instantiation
        # We bind to the schemas of the source/target layers immediately.
        self.src = ForeignKeySchema(
            "src",
            source.schema,
            adjacency_conf=AdjacencyListConf(True, source.schema.index_spec.dtype),
        )
        self.tgt = ForeignKeySchema(
            "tgt",
            target.schema,
            adjacency_conf=AdjacencyListConf(True, target.schema.index_spec.dtype),
        )

        self.schema.register_new_column(self.src)
        self.schema.register_new_column(self.tgt)

    def get_touching(
        self, node_idx: int, kind: Literal["incoming", "outgoing", "both"] = "both"
    ) -> list[int]:
        """Returns a list of edge indices in this layer connected to the given node."""
        if not self.overlay._db:
            raise RuntimeError("DB not initialized")

        db = self.overlay._db
        edges_tbl = db.get_table(self.name)
        missing = edges_tbl.schema.index_spec.missing
        indices = set()

        # Outgoing edges (Node is Source)
        if kind == "outgoing" or kind == "both":
            src_tbl = db.get_table(self.src.target_table.name)
            if node_idx >= 0 and node_idx < len(src_tbl):
                curr = src_tbl[self.src.adj_head].view[node_idx]
                while curr != missing:
                    indices.add(curr)
                    curr = edges_tbl[self.src.adj_next].view[curr]

        # Incoming edges (Node is Target)
        if kind == "incoming" or kind == "both":
            tgt_tbl = db.get_table(self.tgt.target_table.name)
            if node_idx >= 0 and node_idx < len(tgt_tbl):
                curr = tgt_tbl[self.tgt.adj_head].view[node_idx]
                while curr != missing:
                    indices.add(curr)
                    curr = edges_tbl[self.tgt.adj_next].view[curr]

        return list(indices)


@dataclass(init=False)
class GraphOverlay(DbOverlay):
    _db: PackedArrayDB | None
    index_dtype: Any

    def __init__(self, index_dtype=np.uint32):
        self.index_dtype = index_dtype
        self.layers: dict[str, BaseGraphLayer] = {}
        self._db = None

    def add_node_layer(self, name: str, features=None) -> NodeLayer:
        layer = NodeLayer(name, self)
        self.layers[name] = layer
        for f in features or []:
            layer.add_feature(f)
        return layer

    def add_edge_layer(
        self, name: str, source: NodeLayer, target: NodeLayer, features=None
    ) -> EdgeLayer:
        layer = EdgeLayer(name, self, source, target)
        self.layers[name] = layer
        for f in features or []:
            layer.add_feature(f)
        return layer

    def register_schema(self, registry: SchemaRegistry):
        # Inject Layer Schemas
        for _, layer in self.layers.items():
            registry.register_table(layer.schema)

        # Allow Features to inject columns into those schemas
        for name, layer in self.layers.items():
            for feat in layer.features:
                feat.on_schema(registry, name)

    def create_hooks(self) -> list[TransactionHook]:
        hooks = []
        for layer in self.layers.values():
            for f in layer.features:
                h = f.create_hook()
                if h:
                    hooks.append(h)
        return hooks

    def bind(self, db):
        self._db = db
