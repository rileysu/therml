pub mod extension;
pub mod iter;
pub mod allowed_unit;
pub mod builder;
pub mod factory;
pub mod generic;
pub mod padded;

use std::sync::Arc;

use crate::helper::{Shape, Stride, Position, Slice, VarArrayCompatible};
use self::extension::{ExtensionProvider, EmptyExtensionProvider};
use self::factory::EngineTensorFactory;
use self::generic::EngineTensorGeneric;
use self::{iter::EngineTensorUnitIterator, allowed_unit::{AllowedArray, AllowedQuant}};
use std::fmt::Debug;
use super::unit::UnitCompatible;

//Unless otherwise specified every function should make as shallow of a copy as possible
pub trait EngineTensor: Debug {
    type Unit: UnitCompatible;

    fn shape(&self) -> &Shape;
    fn get(&self, pos: &Position) -> Self::Unit;
    fn iter_units(&self) -> EngineTensorUnitIterator<'_, Self::Unit>;

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = Self::Unit>>;
    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>>;

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_>;
}

impl<T: UnitCompatible> PartialEq for dyn EngineTensor<Unit = T> + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.iter_units().zip(other.iter_units()).all(|(a, b)| a == b)
    }
}

#[derive(Debug)]
pub struct Array<T: AllowedArray> {
    stride: Stride,
    shape: Shape,
    data: Arc<[T]>,
    offset: usize,
}

impl<T: AllowedArray> Array<T> {

    pub fn from_data(stride: Stride, shape: Shape, data: Arc<[T]>, offset: usize) -> Self {
        Self {
            stride,
            shape,
            data,
            offset,
        }
    }

    //Am I dumb?
    //This is wrong!!!
    //TODO
    fn is_contiguous(&self) -> bool {
        let mut check: Option<usize> = None;

        for curr in self.stride.iter() {
            match check {
                Some(prev) => {
                    if prev * prev == curr {
                        check = Some(curr);
                    } else {
                        return false;
                    }
                },
                None => {
                    check = Some(curr);
                },
            }
        }

        return true;
    }
}

impl<T: AllowedArray> EngineTensor for Array<T> {
    type Unit = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn get(&self, pos: &Position) -> T {
        let index = pos.tensor_index(&self.stride).unwrap() + self.offset;

        *self.data.as_ref().get(index).unwrap()
    }

    fn iter_units(&self) -> EngineTensorUnitIterator<'_, T> {
        EngineTensorUnitIterator::new(self)
    }

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::new(Self {
            stride: self.stride.clone() ,
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset: self.offset,
        })
    }

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = T>> {
        let offset = slice.start().tensor_index(&self.stride).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: slice.inferred_shape(self.shape()),
            data: self.data.clone(),
            offset,
        })
    }

    //Attempts to efficiently reuse memory if tensor is contiguous
    //If this is not an option it will copy from an iterator
    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = T>> {
        if shape.elements() == self.shape().elements() {
            if self.is_contiguous() {
                Box::new(Array::<T> {
                    stride: Stride::default_from_shape(shape),
                    shape: shape.clone(),
                    data: self.data.clone(),
                    offset: self.offset,
                })
            } else {
                Array::<T>::from_iter(self.iter_units(), shape.clone()).generic()
            }
        } else {
            todo!()
        }
    }

    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        if pos <= self.shape().len() {
            let mut shape_buffer = self.shape().as_slice().to_vec();
            shape_buffer.splice(pos..pos, sub.iter().copied());

            let broadcast_shape = Shape::new(shape_buffer.as_slice().into());

            let mut stride_buffer = self.stride.as_slice().to_vec();
            stride_buffer.splice(pos..pos, std::iter::repeat(0).take(sub.len()));

            let broadcast_stride = Stride::new(stride_buffer.as_slice().into());

            Box::new(Self {
                stride: broadcast_stride,
                shape: broadcast_shape,
                data: self.data.clone(),
                offset: self.offset,
            })
        } else {
            todo!()
        }
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

    fn get(&self, pos: &Position) -> T {
        let index = pos.tensor_index(&self.stride).unwrap() + self.offset;

        *self.data.as_ref().get(index).unwrap()
    }

    fn iter_units(&self) -> EngineTensorUnitIterator<'_, T> {
        EngineTensorUnitIterator::new(self)
    }

    fn clone(&self) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        Box::new(Self {
            stride: self.stride.clone() ,
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset: self.offset,
        })
    }

    fn slice(&self, slice: &Slice) -> Box<dyn EngineTensor<Unit = T>> {
        let offset = slice.start().tensor_index(&self.stride).unwrap();

        Box::from(Self {
            stride: self.stride.clone(),
            shape: self.shape.clone(),
            data: self.data.clone(),
            offset,
        })
    }

    fn reshape(&self, shape: &Shape) -> Box<dyn EngineTensor<Unit = T>> {
        todo!()
    }

    fn broadcast_splice(&self, pos: usize, sub: &[usize]) -> Box<dyn EngineTensor<Unit = Self::Unit>> {
        if pos <= self.shape().len() {
            let mut shape_buffer = self.shape().as_slice().to_vec();
            shape_buffer.splice(pos..pos, sub.iter().copied());

            let broadcast_shape = Shape::new(shape_buffer.as_slice().into());

            let mut stride_buffer = self.stride.as_slice().to_vec();
            stride_buffer.splice(pos..pos, std::iter::repeat(0).take(sub.len()));

            let broadcast_stride = Stride::new(stride_buffer.as_slice().into());

            Box::new(Self {
                stride: broadcast_stride,
                shape: broadcast_shape,
                data: self.data.clone(),
                offset: self.offset,
            })
        } else {
            todo!()
        }
    }

    fn extensions(&self) -> Box<dyn ExtensionProvider + '_> {
        Box::from(EmptyExtensionProvider::from(self))
    }
}