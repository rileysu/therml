pub mod iter;
pub mod basic;
pub mod allowed_unit;
pub mod factory;

use thiserror::Error;
use std::sync::Arc;
use crate::helper::{Shape, Stride, Position, Slice};
use self::{iter::EngineTensorIterator, allowed_unit::AllowedUnit};

#[derive(Debug)]
pub enum EngineTensorSpecs<T: Sized + Copy> {
    Array {
        data: Arc<[T]>,
        offset: usize,
    },
    Quant {
        data: Arc<[T]>,
        offset: usize,
    },
}

#[derive(Debug)]
pub struct EngineTensor<T: AllowedUnit> {
    specs: EngineTensorSpecs<T>,
    stride: Stride,
    shape: Shape,
}

impl<T: AllowedUnit> EngineTensor<T> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub fn get(&self, pos: &Position) -> T {
        match &self.specs {
            EngineTensorSpecs::Array { data, offset} => {
                let index = pos.index(&self.stride).unwrap() + offset;

                *data.as_ref().get(index).unwrap()
            },
            EngineTensorSpecs::Quant { .. } => {
                unimplemented!()
            },
        }
    }

    pub fn iter(&self) -> EngineTensorIterator<'_, T> {
        EngineTensorIterator::new(self)
    }

    pub fn slice(&self, slice: &Slice) -> Self {
        let offset = slice.start().index(self.stride()).unwrap();

        match &self.specs {
            EngineTensorSpecs::Array { data, .. } => Self {
                specs: EngineTensorSpecs::Array { data: data.clone(), offset },
                stride: self.stride.clone(),
                shape: self.shape.clone(),
            },
            EngineTensorSpecs::Quant { data, .. } => Self {
                specs: EngineTensorSpecs::Quant { data: data.clone(), offset },
                stride: self.stride.clone(),
                shape: self.shape.clone(),
            },
        }
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        todo!()
    }
}

