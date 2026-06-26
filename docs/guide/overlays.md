# The Overlays Pattern

As your application grows, directly interacting with `TableSchema`, `DataColSchema`, and `ForeignKeySchema` objects can become cumbersome if your data model has highly specific rules.

To solve this, `packed_data_structures` encourages the use of the **Overlays Pattern**. Overlays are an optional architectural pattern where you build domain-specific wrappers around the core relational database. They expose a constrained, high-level API while managing the underlying tables and schemas internally.

## Example: Graph Overlay

The package includes a built-in `GraphOverlay` as a prime example of this pattern. A graph is essentially just a relational database consisting of "Node" tables and "Edge" tables.

Instead of having to manually instantiate `TableSchema` objects and carefully wire up foreign key colums to build directed edges, the `GraphOverlay` abstracts this away:

```python
from packed_data_structures.graph.overlay import GraphOverlay

# Initialize a domain-specific overlay
graph = GraphOverlay(index_dtype=np.uint32)

# Create a node layer (internally creates a TableSchema)
nodes = graph.add_node_layer("cities")

# Create an edge layer (internally creates a TableSchema + Foreign Keys)
edges = graph.add_edge_layer("roads", source=nodes, target=nodes)
```

### Encapsulating Complexity

When `add_edge_layer` is called, the overlay automatically does the heavy lifting:

1. It creates a new `TableSchema` for the edges.

2. It instantiates two `ForeignKeySchema` objects (`src` and `tgt`) pointing to the source and target node tables.

3. It configures the `AdjacencyListConf` to automatically generate `adj_count` columns.


## When to Build an Overlay

You should consider building an overlay when:

1. You have a rigid domain model (e.g., an Entity-Component System for a game, a Scene Graph, or a Physics simulation).

2. You want to hide the raw database API and expose domain-specific terminology (like `add_node` instead of `add_entry`).

3. You need to enforce specific schema constraints automatically (like ensuring every "Edge" table has exactly two foreign keys).

By treating the raw tables as a backend and your overlay as the frontend, you combine the strict memory efficiency of a packed structure of arrays with the ergonomic DX of a domain-specific API.
