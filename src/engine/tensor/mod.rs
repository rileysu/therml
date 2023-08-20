pub mod array;
pub mod iter;

use thiserror::Error;
use crate::helper::{Shape, Stride, Position, Slice};
use self::iter::EngineTensorIterator;

pub trait EngineTensorConstruct<T: Sized + Copy>
{
    fn from_iter(iter: &mut dyn Iterator<Item = T>, shape: Shape) -> Self;
    fn from_slice(data: &[T], shape: Shape) -> Self;
}

pub trait EngineTensorAccess<T: Sized + Copy>
{
    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;

    fn get(&self, pos: &Position) -> T;

    // Lifetimes here should not be required but they are needed to compile
    fn iter_between<'a>(&'a self, start: &Position, finish: &Position) -> Box<dyn Iterator<Item = T> + 'a> 
    where 
        Self: Sized, 
        T: 'a {
        Box::new(EngineTensorIterator::new(self, start, finish))
    }
}

pub trait EngineTensorFork<T: Sized + Copy> {
    fn slice(&self, slice: Slice) -> Self;
    fn reshape(&self) -> Self;
}

// Base trait for all tensors
pub trait EngineTensor<T: Sized + Copy>: EngineTensorConstruct<T> + EngineTensorAccess<T> + EngineTensorFork<T> {}

//TODO
#[derive(Error, Debug)]
pub enum EngineTensorError {
    #[error("data of length {0} does not match the shape length of {1} in tesnor construction")]
    DataShapeMismatch(usize, usize),
}