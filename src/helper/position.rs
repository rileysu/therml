use std::ops::{Index, IndexMut};
use super::{Stride, Shape};
use thiserror::Error;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Position(Box<[usize]>);

impl Position {
    pub fn new(data: Box<[usize]>) -> Self {
        Self(data)
    }

    pub fn as_boxed_slice(&self) -> &Box<[usize]> {
        &self.0
    }

    pub fn as_mut_boxed_slice(&mut self) -> &mut Box<[usize]> {
        &mut self.0
    }

    pub fn tensor_index(&self, stride: &Stride) -> Result<usize, PositionError> {
        let position_length = stride.as_boxed_slice().len();
        let stride_length = stride.as_boxed_slice().len();

        if position_length == stride_length {
            Ok(self.as_boxed_slice().iter().zip(stride.as_boxed_slice().iter()).map(|(p, s)| p * s).sum())
        } else {
            Err(PositionError::StrideLengthMismatch(position_length, stride_length))
        }
    }

    //Increments or decrements according to the order of positions
    //Euclidian division is used since we need positive elements in a position
    //The math works out...
    pub fn incdec_mut(&mut self, bounds: &Shape, off: i64) {
        let mut curr = off;
        
        for i in (0..bounds.dims()).rev() {
            let signed_bound = bounds[i] as i64;

            curr += self[i] as i64;
            self[i] = curr.rem_euclid(signed_bound) as usize;
            curr = curr.div_euclid(signed_bound);
        }
    }

    pub fn incdec(&self, bounds: &Shape, off: i64) -> Self {
        let mut new_position = self.clone();

        new_position.incdec_mut(bounds, off);

        new_position
    }

    pub fn within_bounds(&self, bounds: &Shape) -> bool {
        self.as_boxed_slice().iter().zip(bounds.as_boxed_slice().iter()).all(|(p, s)| p < s)
    }
}

impl From<&[usize]> for Position {
    fn from(value: &[usize]) -> Self {
        Self(Box::from(value))
    }
}

impl Index<usize> for Position {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_boxed_slice()[index]
    }
}

impl IndexMut<usize> for Position {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_boxed_slice()[index]
    }
}

#[derive(Error, Debug)]
pub enum PositionError {
    #[error("stride length was: {0}, expected: {1}")]
    StrideLengthMismatch(usize, usize)
}

mod test {
    use super::*;

    #[test]
    fn incdec() {
        let shapes = [
            Shape::from([1].as_slice()), 
            Shape::from([3].as_slice()), 
            Shape::from([3, 5, 9].as_slice()),
            Shape::from([9, 5, 3].as_slice()),
            Shape::from([5, 9, 3].as_slice())
        ];

        for shape in shapes.iter() {
            let zero = Position::from(vec![0; shape.as_boxed_slice().len()].as_slice());
            let mut curr = zero.clone();

            for off in 1..shape.len() {
                let max_reps = shape.len() / off;

                for _ in 0..max_reps {
                    curr.incdec_mut(shape, off as i64);
                    assert!(curr.within_bounds(&shape));
                }

                for _ in 0..max_reps {
                    curr.incdec_mut(shape, -(off as i64));
                    assert!(curr.within_bounds(&shape));
                }
            }

            assert_eq!(curr, zero);
        }
    }
}