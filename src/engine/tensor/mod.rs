//pub mod array;
//pub mod iter;

pub mod basic;

use thiserror::Error;
use std::sync::Arc;
use crate::helper::{Shape, Stride, Position, Slice};

pub enum EngineTensorKind {
    Array,
    Quant,
}

pub enum EngineTensorSpec<T: Sized + Copy> {
    Array {
        data: Arc<[T]>,
        offset: usize,
    },
    Quant {
        data: Arc<[T]>,
        offset: usize,
    },
}

pub struct EngineTensor<T: Sized + Copy> {
    spec: EngineTensorSpec<T>,
    stride: Stride,
    shape: Shape,
}

impl<T: Sized + Copy> EngineTensor<T> {
    pub fn from_iter(kind: EngineTensorKind, iter: &mut dyn Iterator<Item = T>, shape: Shape) -> Self {
        match kind {
            EngineTensorKind::Array => Self {
                spec: EngineTensorSpec::Array {
                    data: iter.collect(),
                    offset: 0,
                }, 
                stride: Stride::from(&shape), 
                shape: shape, 
            },
            EngineTensorKind::Quant => Self {
                spec: EngineTensorSpec::Quant {
                    data: iter.collect(),
                    offset: 0,
                }, 
                stride: Stride::from(&shape), 
                shape: shape, 
            },
        }
    }

    pub fn from_slice(kind: EngineTensorKind, data: &[T], shape: Shape) -> Self {
        match kind {
            EngineTensorKind::Array => Self {
                spec: EngineTensorSpec::Array {
                    data: Arc::from(data),
                    offset: 0,
                }, 
                stride: Stride::from(&shape), 
                shape: shape, 
            },
            EngineTensorKind::Quant => Self {
                spec: EngineTensorSpec::Quant {
                     data: Arc::from(data),
                     offset: 0,
                }, 
                stride: Stride::from(&shape), 
                shape: shape,
            },
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub fn get(&self, pos: &Position) -> T {
        match &self.spec {
            EngineTensorSpec::Array { data, offset} => {
                let index = pos.index(&self.stride).unwrap() + offset;

                *data.as_ref().get(index).unwrap()
            },
            EngineTensorSpec::Quant { .. } => {
                todo!()
            },
        }
    }

    pub fn slice(&self, slice: Slice) -> Self {
        let offset = slice.start().index(self.stride()).unwrap();

        match &self.spec {
            EngineTensorSpec::Array { data, .. } => Self {
                spec: EngineTensorSpec::Array { data: data.clone(), offset },
                stride: self.stride.clone(),
                shape: self.shape.clone(),
            },
            EngineTensorSpec::Quant { data, .. } => Self {
                spec: EngineTensorSpec::Quant { data: data.clone(), offset },
                stride: self.stride.clone(),
                shape: self.shape.clone(),
            },
        }
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        todo!()
    }
}

/*
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
*/

//TODO
#[derive(Error, Debug)]
pub enum EngineTensorError {
    #[error("data of length {0} does not match the shape length of {1} in tesnor construction")]
    DataShapeMismatch(usize, usize),
}