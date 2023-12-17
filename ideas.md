# Ideas

## Context

- Context creates tensors and handles data storage of tensors
- Internally will create computation graphs for operations
- Will handle gradient calc

## Engine Tensors
 
- Should handle core utility operations of a tensor (not math operations)
- New tensor instead of inplace as tensors are assumed to always be immutable
- Handles allocation and copying of memory since this can be implementation specific (Might add something to make this easier to implement)

## Engine

- Handles math operations and the method that is used to calculate this for all possible tensor types

## Position, Slice, Indexing

- Tensors all have an implicit ordering of position indexes which follows the significance from left being most significant and right being least significant
- Iterators between positions, reshaping all operate on this idea

## TODO

- Refactor comp_graph to improve errors (ones with no nodekey) and reduce repeated code
- Need a way to remove graphs / subgraphs once they are finished 
    - Maybe seperate graphs from context and create a new graph per training / inference iteration
    - Use phantom to make tensors references to graph so it can't outlive graph
- Probably remove the distinction between context and comp_graph
    - Dump graph on calculation
    - Allow for recalc maybe (as in mutating tensors within the graph)
- Model how the graph interface should look externally as ergonomics is an issue rn