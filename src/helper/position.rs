use super::{Stride, Shape, VarArray, VarArrayCompatible};
use thiserror::Error;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Position(VarArray);

impl Position {
    //Could be interpreted as implementation specific and may not belong in position
    pub fn tensor_index(&self, stride: &Stride) -> Result<usize, PositionError> {
        let position_length = stride.len();
        let stride_length = stride.len();

        if position_length == stride_length {
            Ok(self.iter().zip(stride.iter()).map(|(p, s)| p * s).sum())
        } else {
            Err(PositionError::StrideLengthMismatch(position_length, stride_length))
        }
    }

    //Increments or decrements according to the order of positions
    //Euclidian division is used since we need positive elements in a position
    //The math works out...
    pub fn incdec_mut(&mut self, bounds: &Shape, off: i64) {
        let mut curr = off;
        
        for i in (0..bounds.len()).rev() {
            let signed_bound = bounds.get(i).unwrap() as i64;

            curr += self.get(i).unwrap() as i64;
            *self.get_mut(i).unwrap() = curr.rem_euclid(signed_bound) as usize;
            curr = curr.div_euclid(signed_bound);
        }
    }

    pub fn incdec(&self, bounds: &Shape, off: i64) -> Self {
        let mut new_position = self.clone();

        new_position.incdec_mut(bounds, off);

        new_position
    }

    pub fn within_bounds(&self, bounds: &Shape) -> bool {
        self.iter().zip(bounds.iter()).all(|(p, s)| p < s)
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