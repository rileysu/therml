use std::sync::Arc;
use crate::helper::{Shape, Stride};
use super::{allowed_unit::{AllowedArray, AllowedQuant}, generic::EngineTensorGeneric, Array, EngineTensor, Quant};

pub trait EngineTensorFactory: EngineTensor + EngineTensorGeneric
where Self: Sized 
{
    fn from_iter(iter: impl Iterator<Item = Self::Unit>, shape: Shape) -> Self;
    fn from_slice(data: &[Self::Unit], shape: Shape) -> Self;
    //fn builder() -> impl EngineTensorBuilder<Unit = Self::Unit>;
}

impl<T: AllowedArray> EngineTensorFactory for Array<T> {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Self {
        Array {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        }
    }

    fn from_slice(data: &[T], shape: Shape) -> Self {
        Array {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        }
    }
}

impl<T: AllowedQuant> EngineTensorFactory for Quant<T> {
    fn from_iter(iter: impl Iterator<Item = T>, shape: Shape) -> Self {
        Quant {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: iter.collect(),
            offset: 0,
        }
    }

    fn from_slice(data: &[T], shape: Shape) -> Self {
        Quant {
            stride: Stride::default_from_shape(&shape), 
            shape, 
            data: Arc::from(data),
            offset: 0,
        }
    }
}