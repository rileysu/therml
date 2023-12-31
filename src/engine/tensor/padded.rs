use crate::helper::{Shape, Stride, Position};
use super::{EngineTensor, allowed_unit::AllowedUnit, factory::EngineTensorFactory, iter::EngineTensorUnitIterator};

pub trait AllowedPadded: AllowedUnit {}
impl<T: AllowedUnit> AllowedPadded for T {}

#[derive(Debug)]
pub struct Padded<T: AllowedPadded> {
    tensor: Box<dyn EngineTensor<Unit = T>>,
    stride: Stride,
    shape: Shape,

    high_padding: Box<[usize]>,
    low_padding: Box<[usize]>,

    padding_val: T,
}

impl<T: AllowedPadded> Padded<T> {
    pub fn pad_from(a: Box<dyn EngineTensor<Unit = T>>, padding: Shape, padding_val: T) -> Self {
        if a.shape().dims() == padding.dims() {
            let shape = Shape::new(a.shape().as_boxed_slice().iter().zip(padding.as_boxed_slice().iter()).map(|(o, p)| o + 2 * p).collect());
            let stride = Stride::from(&shape);

            let high_padding = a.shape().as_boxed_slice().iter().zip(padding.as_boxed_slice().iter()).map(|(o, p)| o + p).collect();
            let low_padding = padding.as_boxed_slice().clone();
            
            Self {
                tensor: a,
                stride,
                shape,

                high_padding,
                low_padding,

                padding_val,
            }
        } else {
            todo!()
        }
    }
}

impl<T: AllowedPadded> EngineTensor for Padded<T> {
    type Unit = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn stride(&self) -> &Stride {
        &self.stride
    }

    fn get(&self, pos: &Position) -> Self::Unit {
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