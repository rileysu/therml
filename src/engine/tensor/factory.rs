use std::{marker::PhantomData, sync::Arc};
use crate::helper::{Shape, Stride};
use super::{allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}, EngineTensor, EngineTensorSpecs};

pub trait EngineTensorFactory
where Self: Sized 
{
    type Unit: AllowedUnit;

    fn from_iter(iter: &mut dyn Iterator<Item = Self::Unit>, shape: Shape) -> EngineTensor<Self::Unit>;
    fn from_slice(data: &[Self::Unit], shape: Shape) -> EngineTensor<Self::Unit>;
}

#[derive(Debug)]
pub struct Array<T: AllowedArray> {
    _phantom: PhantomData<T>,
}

#[derive(Debug)]
pub struct Quant<T: AllowedQuant> {
    _phantom: PhantomData<T>,
}

impl<T: AllowedArray> EngineTensorFactory for Array<T> {
    type Unit = T;

    fn from_iter(iter: &mut dyn Iterator<Item = Self::Unit>, shape: Shape) -> EngineTensor<Self::Unit> {
        EngineTensor {
                specs: EngineTensorSpecs::Array {
                data: iter.collect(),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }

    fn from_slice(data: &[Self::Unit], shape: Shape) -> EngineTensor<Self::Unit> {
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

impl<T: AllowedQuant> EngineTensorFactory for Quant<T> {
    type Unit = T;

    fn from_iter(iter: &mut dyn Iterator<Item = Self::Unit>, shape: Shape) -> EngineTensor<Self::Unit> {
        EngineTensor {
            specs: EngineTensorSpecs::Quant {
                data: iter.collect(),
                offset: 0,
            }, 
            stride: Stride::from(&shape), 
            shape: shape, 
        }
    }

    fn from_slice(data: &[Self::Unit], shape: Shape) -> EngineTensor<Self::Unit> {
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