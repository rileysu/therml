# Ideas

## Context

- Context creates tensors and handles data storage of tensors
- Internally will create graphs for operations
- Will handle gradient calc

## Engine Tensors

- Should handle construction, access and forking operations
- New tensor instead of inplace and tensors are immutable
- Handles allocation and copying of memory since this can be implementation specific (Might add something to make this easier to implement)

## Engine

- Should be an abstraction for raw tensors that can be interpreted as tensors of a certain type. Abstraction should be independent from tesnor implementation

## Position, Slice, Indexing

- Tensors all have an implicit ordering of position indexes which follows the significance from left being most significant and right being least significant
- Iterators between positions, reshaping all operate on this idea