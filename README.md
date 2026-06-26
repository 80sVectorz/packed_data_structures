# Packed Data Structures
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](https://80svectorz.github.io/packed_data_structures/)

A flexible, Data-Oriented, swap-and-pop based database system that grants direct memory access to the underlying data structures.

## Overview

This package was originally designed for a usecase that required highly dynamic, scalable graphs. It acts as a flexible, Data-Oriented, swap-and-pop based database system that grants direct memory access to the underlying data structures.

### The Key Trade-off

The fundamental design philosophy of this package centers around one key trade-off: sacrificing explict element ordering in order to support extremely fast edits.

By using a Structure of Arrays (SoA) layout backed by contiguous NumPy arrays, the system achieves excellent cache locality for bulk processing. To maintain this contiguity during row deletions without suffering from $O(N)$ memory shifts, the package relies on an $O(1)$ swap-and-pop mechanic. This means that as you edit your data, the physical ordering of your elements will change.

### Combating the von Neumann Bottleneck

It is well known that graph structures are inherently hostile to modern hardware thanks to the von Neumann bottleneck. Traversing a graph often means jumping wildly around in memory, causing frequent CPU cache misses.

While no system can entirely eliminate this physical reality, `packed_data_structures` attempts to be as optimized as reasonably possible. By packing properties into contiguous columns and automating adjacency list generation, it minimizes the memory footprint of traversing edges.

### Flexibility vs. Performance

This package is designed to support a range of usage styles. It provides high-level abstractions (like the Overlays system and automatic Schema mapping) to make building complex topologies ergonomic.

However, when maximum execution speed is required, **ease of use can be traded in for performance**. You always have the option to bypass the high-level Python abstractions, extract the raw underlying NumPy arrays via the `.view` property, and write highly verbose, complex, but blazingly fast vectorized operations or Numba-JIT compiled kernels.

## Documentation

Full documentation, including a complete First-Principles Guide and API Reference, is available at:
[https://80svectorz.github.io/packed_data_structures/](https://80svectorz.github.io/packed_data_structures/)
