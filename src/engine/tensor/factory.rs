use std::sync::Arc;
use crate::helper::{Shape, Stride};
use super::{allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}, EngineTensor, Array, Quant};

pub trait EngineTensorFactory<T: AllowedUnit>
where Self: Sized 
{
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Box<dyn EngineTensor<T>>;
    fn from_slice(data: &[T], shape: Shape) -> Box<dyn EngineTensor<T>>;
}

impl<T: AllowedArray> EngineTensorFactory<T> for Array<T> {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Box<dyn EngineTensor<T>> {
        Box::from(Array {
            stride: Stride::from(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        })
    }

    fn from_slice(data: &[T], shape: Shape) -> Box<dyn EngineTensor<T>> {
        Box::from(Array {
            stride: Stride::from(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        })
    }
}

impl<T: AllowedQuant> EngineTensorFactory<T> for Quant<T> {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Box<dyn EngineTensor<T>> {
        Box::from(Quant {
            stride: Stride::from(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        })
    }

    fn from_slice(data: &[T], shape: Shape) -> Box<dyn EngineTensor<T>> {
        Box::from(Quant {
            stride: Stride::from(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        })
    }
}