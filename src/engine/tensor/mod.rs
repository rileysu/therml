pub mod iter;
pub mod basic;
pub mod block;
pub mod allowed_unit;
pub mod factory;

use std::sync::Arc;
use crate::helper::{Shape, Stride, Position, Slice};
use self::{iter::EngineTensorIterator, allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}};
use std::fmt::Debug;

pub trait EngineTensor<T: AllowedUnit>: Debug {
    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;
    fn get(&self, pos: &Position) -> T;
    fn iter(&self) -> EngineTensorIterator<'_, T>;
    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<T>>;
    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<T>>;
}

impl<T: AllowedUnit> PartialEq for dyn EngineTensor<T> + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

#[derive(Debug)]
pub struct Array<T: AllowedArray> {
    stride: Stride,
    shape: Shape,
    data: Arc<[T]>,
    offset: usize,
}

impl<T: AllowedArray> EngineTensor<T> for Array<T> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn stride(&self) -> &Stride {
        &self.stride
    }

    fn get(&self, pos: &Position) -> T {
        let index = pos.index(&self.stride).unwrap() + self.offset;

        *self.data.as_ref().get(index).unwrap()
    }

    fn iter(&self) -> EngineTensorIterator<'_, T> {
        EngineTensorIterator::new(self)
    }

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<T>> {
        let offset = slice.start().index(self.stride()).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<T>> {
        todo!()
    }
}

#[derive(Debug)]
pub struct Quant<T: AllowedQuant> {
    stride: Stride,
    shape: Shape,
    data: Arc<[T]>,
    offset: usize,
}

impl<T: AllowedQuant> EngineTensor<T> for Quant<T> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn stride(&self) -> &Stride {
        &self.stride
    }

    fn get(&self, pos: &Position) -> T {
        let index = pos.index(&self.stride).unwrap() + self.offset;

        *self.data.as_ref().get(index).unwrap()
    }

    fn iter(&self) -> EngineTensorIterator<'_, T> {
        EngineTensorIterator::new(self)
    }

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<T>> {
        let offset = slice.start().index(self.stride()).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<T>> {
        todo!()
    }
}
/*#[derive(Debug)]
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

impl<T: AllowedUnit> PartialEq for EngineTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<T: AllowedUnit> Eq for EngineTensor<T> {}*/

