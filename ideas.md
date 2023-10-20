# Ideas

## Context

- Context creates tensors and handles data storage of tensors
- Internally will create graphs for operations
- Will handle gradient calc

## Engine Tensors
 
- Should handle core operations of a tensor (not math operations)
- New tensor instead of inplace as tensors are assumed to always be immutable
- Handles allocation and copying of memory since this can be implementation specific (Might add something to make this easier to implement)

## Engine

- Handles math operations and the method that is used to calculate this for all possible tensor types

## Position, Slice, Indexing

- Tensors all have an implicit ordering of position indexes which follows the significance from left being most significant and right being least significant
- Iterators between positions, reshaping all operate on this idea

## TODO

- Context can have a default engine with which operations will default to and the ability to pass an engine explicitly to change behaviour

- Might need to rework the generics in engine as this seems not straight forward