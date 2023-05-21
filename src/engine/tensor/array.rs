use std::sync::Arc;
use crate::helper::{Shape, Stride};
use super::EngineTensor;

struct ArrayEngineTensor<T: Sized> {
    data: Arc<[T]>,
    shape: Shape,
    stride: Stride,
}



impl<T: Sized> EngineTensor<T> for ArrayEngineTensor<T> {
    type EngineTensorIterator = ArrayEngineTensorIterator<T>;

    fn shape(&self) -> Shape {
        todo!()
    }

    fn stride(&self) -> Stride {
        todo!()
    }

    fn view(&self, shape: Shape) -> Result<Self, ()> {
        todo!()
    }

    fn iter(&self, comp_range: &[std::ops::Range<usize>]) -> Self::EngineTensorIterator {
        todo!()
    }
}

struct ArrayEngineTensorIterator<T: Sized> {
    base: ArrayEngineTensor<T>,
    pos: Vec<usize>,
    end: Vec<usize>,
}

impl<T: Sized> Iterator for ArrayEngineTensorIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        
    }
}