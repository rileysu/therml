use crate::helper::{Position, Shape, VarArrayCompatible};

use super::Slice;

pub struct Iter<'a> {
    pos: Position,
    until: Position,
    slice_shape: Shape,
    starts: Position,
    slice: &'a Slice,
    is_done: bool,
}

impl<'a> Iter<'a> {
    pub fn new(slice: &'a Slice) -> Iter<'a> {
        let slice_shape = slice.inferred_shape();

        Self {
            pos: slice_shape.first(), 
            until: slice_shape.last(),
            slice_shape,
            starts: slice.start(), 
            slice,
            is_done: false,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            None
        } else if self.pos == self.until {
            self.is_done = true;

            Some(self.pos.add(&self.starts).unwrap())
        } else {
            let out = self.pos.add(&self.starts).unwrap();

            self.pos.incdec_mut(&self.slice_shape, 1).unwrap();

            Some(out)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::helper::{position, shape, Interval};

    use super::*;

    #[test]
    fn basic_intervals() {
        let base_shape = shape![6, 5, 4, 3, 2];
        let intervals = Box::from([
            Interval::all(),
            Interval::between(2, 4),
            Interval::start_to(2),
            Interval::end_from(1),
            Interval::only(1),
        ]);

        let slice = Slice::new(intervals, base_shape);

        println!("{:?}", slice.start());
        println!("{:?}", slice.last());

        let ref_positions = [
            position![0, 2, 0, 1, 1],
            position![0, 2, 0, 2, 1],
            position![0, 2, 1, 1, 1],
            position![0, 2, 1, 2, 1],
            position![0, 3, 0, 1, 1],
            position![0, 3, 0, 2, 1],
            position![0, 3, 1, 1, 1],
            position![0, 3, 1, 2, 1],
            position![1, 2, 0, 1, 1],
            position![1, 2, 0, 2, 1],
            position![1, 2, 1, 1, 1],
            position![1, 2, 1, 2, 1],
            position![1, 3, 0, 1, 1],
            position![1, 3, 0, 2, 1],
            position![1, 3, 1, 1, 1],
            position![1, 3, 1, 2, 1],
            position![2, 2, 0, 1, 1],
            position![2, 2, 0, 2, 1],
            position![2, 2, 1, 1, 1],
            position![2, 2, 1, 2, 1],
            position![2, 3, 0, 1, 1],
            position![2, 3, 0, 2, 1],
            position![2, 3, 1, 1, 1],
            position![2, 3, 1, 2, 1],
            position![3, 2, 0, 1, 1],
            position![3, 2, 0, 2, 1],
            position![3, 2, 1, 1, 1],
            position![3, 2, 1, 2, 1],
            position![3, 3, 0, 1, 1],
            position![3, 3, 0, 2, 1],
            position![3, 3, 1, 1, 1],
            position![3, 3, 1, 2, 1],
            position![4, 2, 0, 1, 1],
            position![4, 2, 0, 2, 1],
            position![4, 2, 1, 1, 1],
            position![4, 2, 1, 2, 1],
            position![4, 3, 0, 1, 1],
            position![4, 3, 0, 2, 1],
            position![4, 3, 1, 1, 1],
            position![4, 3, 1, 2, 1],
            position![5, 2, 0, 1, 1],
            position![5, 2, 0, 2, 1],
            position![5, 2, 1, 1, 1],
            position![5, 2, 1, 2, 1],
            position![5, 3, 0, 1, 1],
            position![5, 3, 0, 2, 1],
            position![5, 3, 1, 1, 1],
            position![5, 3, 1, 2, 1],
        ];

        assert_eq!(slice.iter().collect::<Vec<Position>>().len(), slice.elements());

        for (e, r) in slice.iter().zip(ref_positions.iter()) {
            assert_eq!(e, *r);
        }
    }
}