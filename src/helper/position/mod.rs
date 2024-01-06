pub mod iter;

use self::iter::Iter;
use super::{Stride, Shape, VarArray, VarArrayCompatible};
use thiserror::Error;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Position(VarArray);

impl Position {
    //Could be interpreted as implementation specific and may not belong in position
    pub fn tensor_index(&self, stride: &Stride) -> Result<usize, PositionError> {
        if self.len() == stride.len() {
            Ok(self.mul(stride).unwrap().iter().sum())
        } else {
            Err(PositionError::StrideLengthMismatch(self.len(), stride.len()))
        }
    }

    //Increments or decrements according to the order of positions
    //Euclidian division is used since we need positive elements in a position
    //The math works out...
    pub fn incdec_mut(&mut self, bounds: &Shape, off: i64) -> Result<(), PositionError> {
        let mut curr = off;
        
        for i in (0..bounds.len()).rev() {
            let signed_bound = bounds.get(i).unwrap() as i64;

            curr += self.get(i).unwrap() as i64;
            *self.get_mut(i).unwrap() = curr.rem_euclid(signed_bound) as usize;
            curr = curr.div_euclid(signed_bound);
        }

        if curr != 0 {
            //Can't use position in error since it's modified
            Err(PositionError::PositionOverUnderFlow(off))
        } else {
            Ok(())
        }
    }

    pub fn incdec(&self, bounds: &Shape, off: i64) -> Result<Self, PositionError> {
        let mut new_position = self.clone();

        new_position.incdec_mut(bounds, off)?;

        Ok(new_position)
    }

    pub fn iter_positions<'a>(&self, until: &'a Position, bounds: &'a Shape) -> Iter<'a> {
        Iter::new(self.clone(), until, bounds)
    }

    pub fn within_bounds(&self, bounds: &Shape) -> bool {
        self.iter().zip(bounds.iter()).all(|(p, s)| p < s) && self.len() > 0
    }
}

impl VarArrayCompatible for Position {
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

impl From<&[usize]> for Position {
    fn from(value: &[usize]) -> Self {
        Self(VarArray::from(value))
    }
}

#[derive(Error, Debug)]
pub enum PositionError {
    #[error("Stride length was: {0}, expected: {1}")]
    StrideLengthMismatch(usize, usize),
    #[error("Got {0} dimensions but expected {1}")]
    DimensionsMismatch(usize, usize),
    #[error("Offset: {0} causes an overflow or underflow")]
    PositionOverUnderFlow(i64),
    #[error("Overflow or underflow for: {0} and {1}")]
    OperationOverUnderFlow(usize, usize),
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
            let zero = Position::from(vec![0; shape.len()].as_slice());
            let mut curr = zero.clone();

            for off in 1..shape.len() {
                let max_reps = shape.len() / off;

                for _ in 0..max_reps {
                    curr.incdec_mut(shape, off as i64).unwrap();
                    assert!(curr.within_bounds(&shape));
                }

                for _ in 0..max_reps {
                    curr.incdec_mut(shape, -(off as i64)).unwrap();
                    assert!(curr.within_bounds(&shape));
                }
            }

            assert_eq!(curr, zero);
        }
    }
}