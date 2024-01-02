use crate::helper::{Shape, Stride, Position, VarArrayCompatible};
use super::{EngineTensor, allowed_unit::AllowedUnit, factory::EngineTensorFactory, iter::EngineTensorUnitIterator};

pub trait AllowedPadded: AllowedUnit {}
impl<T: AllowedUnit> AllowedPadded for T {}

#[derive(Debug)]
pub struct Padded<T: AllowedPadded> {
    tensor: Box<dyn EngineTensor<Unit = T>>,

    shape: Shape,
    steps: Box<[usize]>,
    start: Position,

    high_padding: Box<[usize]>,
    low_padding: Box<[usize]>,

    padding_val: T,
}

impl<T: AllowedPadded> Padded<T> {
    pub fn pad_from(a: Box<dyn EngineTensor<Unit = T>>, padding: Shape, padding_val: T) -> Self {
        if a.shape().len() == padding.len() {
            let shape = Shape::new(a.shape().iter().zip(padding.iter()).map(|(o, p)| o + 2 * p).collect());
            let steps = shape.clone();
            let start = shape.first();

            let high_padding = a.shape().iter().zip(padding.iter()).map(|(o, p)| o + p).collect();
            let low_padding = padding.clone();
            
            Self {
                tensor: a,

                shape,
                steps,
                start,

                high_padding,
                low_padding,

                padding_val,
            }
        } else {
            todo!()
        }
    }

    fn relative_to_absolute_pos(&self, rel_pos: &Position) -> Position {
        //TODO add errors
       self.start.add(rel_pos).unwrap()
    }
}

impl<T: AllowedPadded> EngineTensor for Padded<T> {
    type Unit = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn get(&self, pos: &Position) -> Self::Unit {
        //Allocation for every get call is poor performance wise...
        let pos = self.relative_to_absolute_pos(pos);

        if pos.within_bounds(self.shape()) {
            let pos_in_unpadded_bounds = pos.as_boxed_slice().iter().zip(self.high_padding.iter()).zip(self.low_padding.iter()).all(|((pos, low), hi)| (*low..*hi).contains(pos));

            if pos_in_unpadded_bounds {
                let middle_pos = Position::new(pos.as_boxed_slice().iter().zip(self.low_padding.iter()).map(|(pos, pad)| pos - pad).collect());

                self.tensor.get(&middle_pos)
            } else {
                self.padding_val
            }
        } else {
            todo!()
        }
    }

    fn iter_unit(&self) -> super::iter::EngineTensorUnitIterator<'_, Self::Unit> {
        EngineTensorUnitIterator::new(self)
    }

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        todo!()
    }

    fn slice(&self, slice: &crate::helper::Slice) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        todo!()
    }

    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        todo!()
    }

    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        todo!()
    }

    fn extensions(&self)-> Box<dyn super::extension::ExtensionProvider + '_> {
        todo!()
    }
    
}