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

## Algorithms

### Conv2d
```
a1: (batches, in_channels, y, x)
k1: (out_channels, in_channels, k_y, k_x)

---

let a2: (batches, out_channels, in_channels, y, x) = a1.broadcast_splice([out_channels], 1)
let k2: (batches, out_channels, in_channels, k_y, k_x) = k1.broadcast_splice([batches], 0)

let k_half_len_y = (k_y // 2)
let k_half_len_x = (k_x // 2)

let start_y = k_half_len_y
let end_y = y - start_y
let start_x = k_half_len_x
let end_x = x - start_x

let mut out_units = 

for curr_y in start_y..end_y {
    for curr_x in start_x..end_x {
        let a3: (batches, out_channels, in_channels, k_y, k_x) = a2.slice([:, :, :, curr_y-k_half_len_y:curr_y+k_half_len_y, curr_x-k_half_len_x:curr_x+k_half_len_x])

        let r1: (batches, out_channels, in_channels, k_y, k_x) = a3 * k2

        let r2: (batches, out_channels, in_channels * k_y * k_x) = r1.reshape([batches, out_channels, in_channels, k_y * k_x])

        let r3: (batches, out_channels, 1) = r2.sum()

        // batches, x, y
        out_units.extend(r3.iter_unit())
    }
}

return tensor::from_slice(&out_units): (y, x, batches, out_channels)


```

## TODO

- Refactor comp_graph to improve errors (ones with no nodekey) and reduce repeated code
- Need a way to remove graphs / subgraphs once they are finished 
    - Maybe seperate graphs from context and create a new graph per training / inference iteration
    - Use phantom to make tensors references to graph so it can't outlive graph
- Probably remove the distinction between context and comp_graph
    - Dump graph on calculation
    - Allow for recalc maybe (as in mutating tensors within the graph)
- Model how the graph interface should look externally as ergonomics is an issue rn
- Make a variable length array to use in position, shape, slice
    - Each can use the underlying type to handle basic operations