use std::sync::Arc;
use crate::helper::{Shape, Stride, Position};
use super::EngineTensor;

pub struct QuantTensor<T: Sized + Copy> {
    data: Arc<[T]>,
    offset: usize,
    shape: Shape,
    stride: Stride,
}

impl<T: Sized + Copy> EngineTensor<T> for QuantTensor<T> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn stride(&self) -> &Stride {
        &self.stride
    }

    fn get(&self, pos: &Position) -> T {
        todo!()
    }

    fn slice(&self) -> Self {
        todo!()
    }
}