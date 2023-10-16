use std::ops::Index;

use crate::helper::{Shape, Position};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(Box<[usize]>);

impl Stride {
    pub fn new(data: Box<[usize]>) -> Self {
        Self(data)
    }

    pub fn as_boxed_slice(&self) -> &Box<[usize]> {
        &self.0
    }

    pub fn get_index(&self, pos: &Position) -> usize {
        self.as_boxed_slice().iter().zip(pos.as_boxed_slice().iter()).map(|(s, p)| s * p).sum()
    }
}

impl From<&[usize]> for Stride {
    fn from(value: &[usize]) -> Self {
        Self(Box::from(value))
    }
}

impl From<&Shape> for Stride {
    fn from(value: &Shape) -> Self {
        let mut stride: Vec<usize> = Vec::with_capacity(value.as_boxed_slice().len());

        let mut next = 1usize;
        for dim in value.as_boxed_slice().iter().rev() {
            stride.push(next);
            next *= dim;
        }

        stride.reverse();

        Stride(stride.into_boxed_slice())
    }
}

impl Index<usize> for Stride {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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
            assert_eq!(Stride::from(&shape), stride);
        }
    }
}