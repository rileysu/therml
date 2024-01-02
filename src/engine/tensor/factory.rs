use std::sync::Arc;
use crate::helper::{Shape, Stride};
use super::{allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}, EngineTensor, Array, Quant};

pub trait EngineTensorFactory
where Self: Sized 
{
    type Unit: AllowedUnit;

    fn from_iter(iter: impl Iterator<Item = Self::Unit>, shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn from_slice(data: &[Self::Unit], shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
}

impl<T: AllowedArray> EngineTensorFactory for Array<T> {
    type Unit = T;

    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(Array {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        })
    }

    fn from_slice(data: &[T], shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(Array {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        })
    }
}

impl<T: AllowedQuant> EngineTensorFactory for Quant<T> {
    type Unit = T;

    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(Quant {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        })
    }

    fn from_slice(data: &[T], shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::from(Quant {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        })
    }
}