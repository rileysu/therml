use std::sync::Arc;
use crate::helper::{Shape, Stride, Position, Slice};
use super::{EngineTensor, EngineTensorConstruct, EngineTensorAccess, EngineTensorFork};

mod basic;

#[derive(Debug)]
pub struct ArrayTensor<T: Sized + Copy> {
    data: Arc<[T]>,
    stride: Stride,
    shape: Shape,
    offset: usize,
}

impl<T: Sized + Copy> ArrayTensor<T> {
    //Get a slice of the active data in the tensor
    pub fn get_data_slice(&self) -> &[T] {
        &self.data[self.offset .. self.offset + self.shape.len()]
    }
}

impl<T: Sized + Copy> EngineTensor<T> for ArrayTensor<T> {}

impl<T: Sized + Copy> EngineTensorConstruct<T> for ArrayTensor<T> {
    fn from_iter(iter: &mut dyn Iterator<Item = T>, shape: Shape) -> Self {
        Self {
            data: iter.collect(),
            stride: Stride::from(&shape),
            shape: shape,
            offset: 0,
        }
    }
    
    fn from_slice(data: &[T], shape: Shape) -> Self {
        Self {
            data: Arc::from(data),
            stride: Stride::from(&shape),
            shape: shape,
            offset: 0,
        }
    }
}


impl<T: Sized + Copy> EngineTensorAccess<T> for ArrayTensor<T> {
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
}

impl<T: Sized + Copy> EngineTensorFork<T> for ArrayTensor<T> {
    // In this case all we should need to do is provide an offset that is the offset of the slice
    // and create a new shape that is the inferred shape of the slice
    // this does not account for step yet
    fn slice(&self, slice: Slice) -> Self {
        let offset = slice.start().index(self.stride()).unwrap();

        Self {
            offset,
            data: self.data.clone(),
            stride: self.stride().clone(),
            shape: slice.inferred_shape(self.shape()),
        }
    }

    fn reshape(&self) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::helper::Interval;

    use super::*;

    #[test]
    fn slice_with_normal_step() {
        let tensor = ArrayTensor::from_slice(&[0, 1, 2, 3, 4, 5], Shape::from([2, 3].as_slice()));

        let sliced_tensor = tensor.slice(Slice::from([Interval::all(), Interval::finish_from(1)].as_slice()));

        println!("{:?}", sliced_tensor);
        println!("{:?}", sliced_tensor.iter_between(&Position::from([0, 0].as_slice()), &Position::from([1, 1].as_slice())).collect::<Vec<i32>>())
    }
}