use std::iter;

use thiserror::Error;

use crate::{engine::unit::UnitCompatible, helper::{slice::iter::SliceIter, Interval, Slice, VarArrayCompatible}};

use super::EngineTensor;

pub struct EngineTensorSubTensorIterator<'a, T: UnitCompatible> {
    tensor: &'a dyn EngineTensor<Unit = T>,
    sub_slice_iter: SliceIter,
    num_all_dims: usize,
}

impl<'a, T: UnitCompatible> EngineTensorSubTensorIterator<'a, T> {
    pub fn new(tensor: &'a dyn EngineTensor<Unit = T>, sub_slice: Slice) -> Result<Self, EngineTensorSubTensorIteratorError> {
        let sub_slice_dims = sub_slice.inferred_shape().len();
        let tensor_dims = tensor.shape().len();

        if sub_slice_dims <= tensor_dims {
            Ok(Self {
                tensor,
                sub_slice_iter: sub_slice.iter(), // :(
                num_all_dims: tensor_dims - sub_slice_dims,
            })
        } else {
            Err(EngineTensorSubTensorIteratorError::SubSliceTooManyDims(sub_slice_dims, tensor_dims))
        }
    }
}

impl<'a, T: UnitCompatible> Iterator for EngineTensorSubTensorIterator<'a, T> {
    type Item = Box<dyn EngineTensor<Unit = T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_pos = self.sub_slice_iter.next();

        next_pos.map(|pos| {
            let intervals = pos.iter().map(|x| Interval::only(x)).chain(iter::repeat(Interval::all()).take(self.num_all_dims)).collect::<Vec<_>>();

            self.tensor.slice(&intervals)
        })
    }
}

#[derive(Debug, Error)]
pub enum EngineTensorSubTensorIteratorError {
    #[error("The sub slice is too many dims the tensor: {0} does not fit in {1}")]
    SubSliceTooManyDims(usize, usize),
}

#[cfg(test)]
mod test {
    use crate::{engine::tensor::{factory::EngineTensorFactory, EngineTensor}, engine_impl::tensor::array::Array, helper::{Shape, shape, Interval, Slice}};

    use super::EngineTensorSubTensorIterator;

    #[test]
    fn basic() {
        let tensor = Array::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape![3, 4]);
        let sub_slice = Slice::new([Interval::all()].into(), tensor.shape().clone());

        let iter = EngineTensorSubTensorIterator::new(&tensor, sub_slice).unwrap();

        for sub in iter {
            println!("{:?}", sub.iter_units().collect::<Vec<_>>());
        }
    }
}