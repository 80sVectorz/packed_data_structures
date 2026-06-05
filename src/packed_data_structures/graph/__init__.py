from __future__ import annotations

import numpy as np
from typing import Any

from packed_data_structures.schemas import IndexSpec, ForeignKeySchema
from packed_data_structures.overlays.base import DbOverlay
from packed_data_structures.overlays.registry import SchemaRegistry
from packed_data_structures.overlays.hooks import TransactionHook


class GraphFeature:
    """Base for components attached to graph layers."""

    def attach(self, layer: BaseGraphLayer):
        pass

    def on_schema(self, registry: SchemaRegistry, layer_name: str):
        pass

    def create_hook(self) -> TransactionHook | None:
        return None


class BaseGraphLayer:
    def __init__(self, name: str, overlay: GraphOverlay):
        self.name = name
        self.overlay = overlay
        self.features: list[GraphFeature] = []

    def add_feature(self, feat: GraphFeature):
        self.features.append(feat)
        feat.attach(self)
        return self

    def add_entry(self, **kwargs) -> int:
        return self.overlay._db.get_table(self.name).add_entry(kwargs)[0]


class NodeLayer(BaseGraphLayer):
    pass


class EdgeLayer(BaseGraphLayer):
    def __init__(
        self, name: str, overlay: GraphOverlay, source: NodeLayer, target: NodeLayer
    ):
        super().__init__(name, overlay)
        self.source = source
        self.target = target


class GraphOverlay(DbOverlay):
    def __init__(self, index_dtype=np.uint32):
        self.index_dtype = index_dtype
        self.layers: dict[str, BaseGraphLayer] = {}
        self._db: Any = None

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

    # --- Overlay Implementation ---

    def register_schema(self, registry: SchemaRegistry):
        spec = IndexSpec.from_dtype(self.index_dtype)

        # Register Core Tables
        for name in self.layers:
            registry.ensure_table(name, spec)

        # Register Structural Columns (FKs) & Features
        for name, layer in self.layers.items():
            if isinstance(layer, EdgeLayer):
                # We reuse the actual TableSchema objects from the registry
                # This ensures the FK points to the real table definition
                src_tbl = registry.ensure_table(layer.source.name, spec)
                dst_tbl = registry.ensure_table(layer.target.name, spec)

                # Reuse existing ForeignKeySchema logic
                registry.add_column(name, ForeignKeySchema("src", src_tbl))
                registry.add_column(name, ForeignKeySchema("dst", dst_tbl))

            # Apply Features
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
