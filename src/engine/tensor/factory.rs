use std::sync::Arc;
use crate::helper::{Shape, Stride};
use super::{allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}, EngineTensor, EngineTensorSpecs};

pub trait EngineTensorFactory<T: AllowedUnit>
where Self: Sized 
{
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> EngineTensor<T>;
    fn from_slice(data: &[T], shape: Shape) -> EngineTensor<T>;
}

#[derive(Debug)]
pub struct Array {}

#[derive(Debug)]
pub struct Quant {}

impl<T: AllowedArray> EngineTensorFactory<T> for Array {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> EngineTensor<T> {
        EngineTensor {
                specs: EngineTensorSpecs::Array {
                data: iter.collect(),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }

    fn from_slice(data: &[T], shape: Shape) -> EngineTensor<T> {
        EngineTensor {
                specs: EngineTensorSpecs::Array {
                data: Arc::from(data),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }
}

impl<T: AllowedQuant> EngineTensorFactory<T> for Quant {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> EngineTensor<T> {
        EngineTensor {
            specs: EngineTensorSpecs::Quant {
                data: iter.collect(),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }

    fn from_slice(data: &[T], shape: Shape) -> EngineTensor<T> {
        EngineTensor {
                specs: EngineTensorSpecs::Quant {
                data: Arc::from(data),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }
}