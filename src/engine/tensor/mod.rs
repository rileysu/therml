pub mod extension;
pub mod iter;
pub mod basic;
pub mod allowed_unit;
pub mod factory;

use std::sync::Arc;

use crate::helper::{Shape, Stride, Position, Slice};
use self::extension::{ExtensionProvider, EmptyExtensionProvider};
use self::{iter::EngineTensorIterator, allowed_unit::{AllowedUnit, AllowedArray, AllowedQuant}};
use std::fmt::Debug;



pub trait EngineTensor<>: Debug {
    type Unit: AllowedUnit;

    fn shape(&self) -> &Shape;
    fn stride(&self) -> &Stride;
    fn get(&self, pos: &Position) -> Self::Unit;
    fn iter(&self) -> EngineTensorIterator<'_, Self::Unit>;
    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn extensions(&self)-> Box<dyn ExtensionProvider + '_>;
}

impl<T: AllowedUnit> PartialEq for dyn EngineTensor<Unit = T> + '_ {
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

impl<T: AllowedArray> EngineTensor for Array<T> {
    type Unit = T;

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

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = T>> {
        let offset = slice.start().index(self.stride()).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<Unit = T>> {
        todo!()
    }

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_> {
        Box::from(EmptyExtensionProvider::from(self))
    }
}

#[derive(Debug)]
pub struct Quant<T: AllowedQuant> {
    stride: Stride,
    shape: Shape,
    data: Arc<[T]>,
    offset: usize,
}

impl<T: AllowedQuant> EngineTensor for Quant<T> {
    type Unit = T;

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

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = T>> {
        let offset = slice.start().index(self.stride()).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: Shape) -> Box<dyn EngineTensor<Unit = T>> {
        todo!()
    }

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_> {
        Box::from(EmptyExtensionProvider::from(self))
    }
}