use std::ops::Index;

use crate::helper::Shape;

use super::{VarArray, VarArrayCompatible, Unit};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(VarArray);

impl Stride {
    pub fn default_from_shape(shape: &Shape) -> Self {
        let mut stride: Vec<usize> = Vec::with_capacity(shape.len());

        let mut next = 1usize;
        for dim in shape.iter().collect::<Vec<Unit>>().iter().rev() {
            stride.push(next);
            next *= dim;
        }

        stride.reverse();

        Stride(stride.as_slice().into())
    }
}

impl VarArrayCompatible for Stride {
    fn new(varr: VarArray) -> Self {
        Self(varr)
    }

    fn vararray(&self) -> &VarArray {
        &self.0
    }

    fn vararray_mut(&mut self) -> &mut VarArray {
        &mut self.0
    }
}

impl From<&[usize]> for Stride {
    fn from(value: &[usize]) -> Self {
        Self(VarArray::from(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_shape_examples() {
        let examples = [
            (Shape::from([1].as_slice()), Stride::from([1].as_slice())),
            (Shape::from([20, 50, 4].as_slice()), Stride::from([200, 4, 1].as_slice()))
        ];

        for (shape, stride) in examples {
            assert_eq!(Stride::default_from_shape(&shape), stride);
        }
    }
}